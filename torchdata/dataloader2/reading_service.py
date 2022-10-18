# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import multiprocessing as mp

from abc import ABC, abstractmethod

from datetime import timedelta
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader

from torchdata._constants import default_timeout_in_s
from torchdata.dataloader2 import communication
from torchdata.dataloader2.graph import DataPipe
from torchdata.datapipes.iter import FullSync, IterableWrapper, IterDataPipe


class ReadingServiceInterface(ABC):
    r"""
    Interface for ``ReadingService``. Please extend custom ``ReadingService`` based on this interface class.

    ReadingService must be picklable prior to ``initialize`` being called. This is because a copy of it will be
    created by ``DataLoader2`` to avoid the situation where the same ReadingService object is used by
    multiple ``DataLoader2``, and its internal state will be modifiable by each of them.

    As a result of this constraint, certain initialization steps may need to take place within the
    ``initialize`` method rather than ``__init__`` of the ReadingService class.
    """

    @abstractmethod
    def initialize(self, datapipe: DataPipe) -> DataPipe:
        r"""
        ``ReadingService`` takes a ``DataPipe`` graph, adapts it into a new ``DataPipe`` graph based on the custom need.
        Called once in creating ``DataLoader2`` iterator at first time. Prior to calling this method,
        the ``ReadingService`` object must be picklable.

        Args:
            datapipe: Original ``DataPipe`` graph.

        Return:
            An adapted or a new ``DataPipe`` graph.
        """
        pass

    def finalize(self) -> None:
        r"""
        ``ReadingService`` cleans up internal states and fully shuts down the service.
        Called in ``DataLoader2``'s ``shutdown`` and ``__del__``.
        """
        pass

    def initialize_iteration(self) -> None:
        r"""
        ``ReadingService`` spins up service for an epoch. Called at the beginning
        of every time getting ``DataLoader2`` iterator.
        """
        pass

    def finalize_iteration(self) -> None:
        r"""
        ``ReadingService`` ends service after an epoch is finished. Called when
        the iterator of ``DataLoader2`` is depleted.
        """
        pass

    def get_datapipe_length(self) -> Optional[int]:
        """
        Called by DataLoader to return the length of the DataPipe. Output should be
        ``None`` if the length is not available for any reason. This can
        vary by multiprocessing/distributed setting, but the output value should
        not depend on whether iteration has begun or not.
        """


class CheckpointableReadingServiceInterface(ReadingServiceInterface):
    r"""
    Extend ``ReadingServiceInterface`` with two additional methods to save/restore the state of the data-processing graph.
    """

    @abstractmethod
    def checkpoint(self) -> bytes:
        """
        ``ReadingService`` serializes the internal states. Called in ``DataLoader2.state_dict``.
        """
        pass

    @abstractmethod
    def restore(self, datapipe: DataPipe, serialized_state: bytes) -> DataPipe:
        """
        ``ReadingService`` adapts ``DataPipe`` graph based on the serialized state.
        Called once in creating ``DataLoader2`` iterator at first time.
        Counterpart of ``initialize``, which adapt ``DataPipe`` graph from scratch.

        Args:
            datapipe: original ``DataPipe`` graph before adapted by ``ReadingService``
            serialized_state: The serialized state of internal state used to restore the state
                of the adapted ``DataPipe`` graph.

        Returns:
            Adapted ``DataPipe`` generated from the serialized state.
        """
        pass


def _collate_no_op(batch):
    return batch[0]


class _IterateQueueDataPipes(IterDataPipe):
    def __init__(self, datapipes):
        # TODO(VitalyFedyunin): Consider combining _IterateQueueDataPipes and QueueWrapper
        # into one class, which supports any number of queues.
        self.datapipes = datapipes
        for dp in self.datapipes:
            if not isinstance(dp, communication.iter.QueueWrapper):
                raise Exception("Source datapipes should be an instance of iter.QueueWrapper")

    def __iter__(self):
        total_pipes = len(self.datapipes)
        disabled_pipe = [False] * len(self.datapipes)
        cnt_disabled_pipes = 0

        for idx in range(total_pipes):
            self.datapipes[idx].protocol.request_next()

        while cnt_disabled_pipes < total_pipes:
            for idx in range(total_pipes):
                if not disabled_pipe[idx]:
                    response = self.datapipes[idx].protocol.get_response_next(block=True)
                    if isinstance(response, communication.messages.StopIterationResponse):
                        disabled_pipe[idx] = True
                        cnt_disabled_pipes += 1
                        continue
                    if isinstance(response, communication.messages.InvalidStateResponse):
                        raise communication.iter.InvalidStateResetRequired
                    if isinstance(response, communication.messages.TerminateResponse):
                        raise communication.iter.TerminateRequired
                    self.datapipes[idx].protocol.request_next()
                    yield response.value

    def reset(self):
        # Collect all existing requests results to clear queues
        for dp in self.datapipes:
            if dp.protocol.waiting_for_response():
                dp.protocol.get_response_next(block=True)
        # NonBlocking DataPipes do not reset automatically, have to do it manually
        for dp in self.datapipes:
            dp.reset_iterator()

    def reset_epoch(self, *args):
        for dp in self.datapipes:
            dp.protocol.discard_existing_request()
        for dp in self.datapipes:
            dp.protocol.request_reset_epoch(*args)


class PrototypeMultiProcessingReadingService(ReadingServiceInterface):
    r"""
    ``PrototypeMultiProcessingReadingService`` that spawns multiple subprocesses to iterate the ``DataPipe`` graph.
    This ``ReadingService`` is still under prototype stage and will replace ``MultiProcessingReadingService`` eventually.

    Args:
        num_workers (int, optional): How many subprocesses to use for data loading.
            ``0`` will be replaced by ``InProcessReadingService`` in the future.
        multiprocessing_context (str, optional): Multiprocessing starting method.
            If method is None then the default context is returned.
            Otherwise method should be 'fork', 'spawn'.
    """
    num_workers: int
    processes: List
    datapipes: List
    combined_datapipes: Optional[IterDataPipe]

    def __init__(
        self,
        num_workers: int = 0,
        multiprocessing_context=None,
        prefetch_worker: int = 10,
        prefetch_mainloop: int = 10,
    ) -> None:
        self.num_workers = num_workers
        # TODO(613): Should be one of 'fork', 'spawn'
        self.multiprocessing_context = multiprocessing_context
        self.prefetch_worker = prefetch_worker
        self.prefetch_mainloop = prefetch_mainloop
        self.processes = []
        self.datapipes = []
        self.combined_datapipes = None
        self._length: Optional[int] = None

    @staticmethod
    def init_datapipe_process(num_workers, worker_id, datapipe):
        # TODO(614): Add distributed support
        # TODO(615): Add shuffle determinism support
        torch.utils.data.graph_settings.apply_sharding(datapipe, num_workers, worker_id)

    @staticmethod
    def call_on_epoch_reset(datapipe, *args):
        # This function will receive worker local copy of datapipe and args value from initialize_iteration
        pass

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        r"""
        ``MultiProcessingReadingService`` finds information about sharding,
        separates graph by multiple pieces and reconnects it using queues.
        creates subprocesses.
        """
        try:
            self._length = len(datapipe)
        except TypeError:
            pass
        if self.num_workers == 0:
            # TODO(616): Warn and recommend usage of InProcessReadingService
            return datapipe

        if self.prefetch_worker > 0:
            datapipe = datapipe.prefetch(self.prefetch_worker)

        for worker_id in range(self.num_workers):
            # TODO(617): Separate into function, because we also need to apply distributed seed
            #            and call it inside process
            call_inside_process = functools.partial(self.init_datapipe_process, self.num_workers, worker_id)
            call_on_epoch_reset = self.call_on_epoch_reset
            ctx = mp.get_context(self.multiprocessing_context)
            (process, req_queue, res_queue) = communication.eventloop.SpawnProcessForDataPipeline(
                ctx,
                datapipe,
                call_inside_process,
                call_on_epoch_reset,
            )
            process.start()
            self.processes.append((process, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
            )
            self.datapipes.append(local_datapipe)

        self.combined_datapipes = _IterateQueueDataPipes(self.datapipes)
        if self.prefetch_mainloop > 0:
            self.combined_datapipes = self.combined_datapipes.prefetch(self.prefetch_mainloop)
        return self.combined_datapipes  # type: ignore[return-value]

    def initialize_iteration(self) -> None:
        if self.combined_datapipes is not None:
            if self.prefetch_mainloop > 0:
                # Stop prefetching first
                self.combined_datapipes.reset()
                self.combined_datapipes.source_datapipe.reset_epoch()
                self.combined_datapipes.source_datapipe.reset()
            else:
                self.combined_datapipes.reset_epoch()
                self.combined_datapipes.reset()

    def __del__(self):
        self.finalize()

    def finalize(self) -> None:
        r"""
        ``MultiProcessingReadingService`` invalidate states & properly exits all subprocesses.
        """
        # TODO(618): Check if anyone stuck with messages
        def clean_me(process, req_queue, res_queue):
            # TODO(619): Can send terminations simultaneously
            # TODO(620): Make termination a function of QueueWrapperDataPipe (similar to reset)
            req_queue.put(communication.messages.TerminateRequest())
            _ = res_queue.get()
            process.join()

        for process, req_queue, res_queue in self.processes:
            clean_me(process, req_queue, res_queue)

        self.processes = []

    def get_datapipe_length(self) -> Optional[int]:
        return self._length


class MultiProcessingReadingService(ReadingServiceInterface):
    r"""
    ``MultiProcessingReadingService`` that utilizes ``torch.utils.data.DataLoader`` to
    launch subprocesses for ``DataPipe`` graph. Please refers to documents of ``DataLoader``
    in https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for all arguments.

    Note:
        This ``ReadingService`` be replaced by :class:`PrototypeMultiProcessingReadingService`.
    """
    num_workers: int
    pin_memory: bool
    timeout: float
    worker_init_fn: Optional[Callable[[int], None]]
    prefetch_factor: int
    persistent_workers: bool

    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        multiprocessing_context=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.dl_: Optional[DataLoader] = None
        self._length: Optional[int] = None

    # Wrap the DataLoader with IterableWrapper to respect type annotation
    def initialize(self, datapipe: DataPipe) -> DataPipe:
        self.dl_ = DataLoader(
            datapipe,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            multiprocessing_context=self.multiprocessing_context,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            # TODO(621): `collate_fn` is necessary until we stop using DLv1 https://github.com/pytorch/data/issues/530
            collate_fn=_collate_no_op,
            batch_size=1,  # This reading service assume batching is done via DataPipe
        )
        try:
            self._length = len(self.dl_)
        except TypeError:
            self._length = None
        return IterableWrapper(self.dl_)  # type: ignore[return-value]

    def finalize(self) -> None:
        if self.persistent_workers and self.dl_ is not None and self.dl_._iterator is not None:
            self.dl_._iterator._shutdown_workers()  # type: ignore[attr-defined]
            self.dl_._iterator = None

    def get_datapipe_length(self) -> Optional[int]:
        return self._length


class DistributedReadingService(ReadingServiceInterface):
    r"""
    ``DistributedReadingSerivce`` handles distributed sharding on the graph of ``DataPipe`` and
    guarantee the randomness by sharing the same seed across the distributed processes.

    Args:
        timeout: Timeout for operations executed against the process group in seconds.
            Default value equals 30 minutes.
    """

    def __init__(self, timeout: int = default_timeout_in_s):
        if not dist.is_available():
            raise RuntimeError("Torch Distributed is required to be available")
        self._world_size: int = 1
        self._rank: int = 0
        self._datapipe: Optional[DataPipe] = None
        self._timeout: int = timeout
        self._pg: Optional[dist.ProcessGroup] = None
        self._length: Optional[int] = None

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        r"""
        Launches the ``gloo``-backend distributed process group. Carries out distributed sharding
        on the graph of ``DataPipe`` and returnes the graph attached with a ``FullSyncIterDataPipe``
        at the end.
        """
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Torch Distributed is required to be initialized")
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._pg = dist.new_group(backend="gloo", timeout=timedelta(seconds=self._timeout))
        torch.utils.data.graph_settings.apply_sharding(
            datapipe,
            self._world_size,
            self._rank,
        )
        # Only append FullSyncIterDataPipe if it's not presented at the end of the pipeline
        if not isinstance(datapipe, FullSync):
            datapipe = datapipe.fullsync(self._timeout)
        self._datapipe = datapipe
        try:
            self._length = len(datapipe)
        except TypeError:
            self._length = None
        return datapipe

    def initialize_iteration(self) -> None:
        r"""
        Shares the same seed from rank 0 to other ranks across the distributed processes
        and apply the random seed to the ``DataPipe`` graph.
        """
        # TODO: Seed Generator should be moved to DataLoader2 after the API
        #       change of initialize_iteration is landed.
        seed = self._share_seed()
        _seed_generator = torch.Generator()
        _seed_generator.manual_seed(seed)
        assert self._datapipe is not None
        self._datapipe = torch.utils.data.graph_settings.apply_random_seed(
            self._datapipe,
            _seed_generator,
        )

    def _share_seed(self):
        shared_seed = torch.empty((), dtype=torch.int64).random_()
        dist.broadcast(shared_seed, src=0, group=self._pg)
        return shared_seed.item()

    def finalize(self) -> None:
        r"""
        Clean up the distributed process group.
        """
        self._pg = None

    def get_datapipe_length(self) -> Optional[int]:
        return self._length
