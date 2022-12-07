# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import multiprocessing as mp

from abc import ABC, abstractmethod
from collections import deque
from datetime import timedelta
from functools import partial
from typing import Callable, Deque, List, Optional

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.grouping import SHARDING_PRIORITIES

from torchdata._constants import default_dl2_worker_join_timeout_in_s, default_timeout_in_s
from torchdata.dataloader2 import communication
from torchdata.dataloader2.graph import DataPipe
from torchdata.dataloader2.utils import generate_random_scalar_tensor, process_init_fn, process_reset_fn, WorkerInfo
from torchdata.dataloader2.utils.worker import _DistInfo
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
    r"""
    Takes in ``QueueWrapper``s and iterates through them in a round-robin manner to get batches one-by-one.

    Typically, each worker has one ``QueueWrapper``.
    """

    def __init__(self, datapipes):
        # TODO(VitalyFedyunin): Consider combining _IterateQueueDataPipes and QueueWrapper
        #                       into one class, which supports any number of queues.
        for dp in datapipes:
            if not isinstance(dp, communication.iter.QueueWrapper):
                raise Exception("Source datapipes should be an instance of iter.QueueWrapper")
        self.datapipes = datapipes
        self.res_buffers: List[Deque] = [deque() for _ in range(len(datapipes))]

    def __iter__(self):
        total_pipes = len(self.datapipes)
        disabled_pipe = [False] * len(self.datapipes)
        cnt_disabled_pipes = 0

        for idx in range(total_pipes):
            self.datapipes[idx].protocol.request_next()

        while cnt_disabled_pipes < total_pipes:
            for idx in range(total_pipes):
                if not disabled_pipe[idx]:
                    # Check if buffer of the DataPipe is empty, if not, yield one before requesting next
                    if len(self.res_buffers[idx]):
                        response = self.res_buffers[idx].popleft()
                    else:
                        response = self.datapipes[idx].protocol.get_response_next(block=True)
                    if isinstance(response, communication.messages.StopIterationResponse):
                        disabled_pipe[idx] = True
                        cnt_disabled_pipes += 1
                        continue
                    if isinstance(response, communication.messages.InvalidStateResponse):
                        raise communication.iter.InvalidStateResetRequired
                    if isinstance(response, communication.messages.TerminateResponse):
                        raise communication.iter.TerminateRequired
                    if len(self.res_buffers[idx]) == 0:  # Only request if buffer is empty
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

    def reset_epoch(self, reset_fn: Callable[[WorkerInfo, DataPipe], DataPipe]):
        for dp in self.datapipes:
            dp.protocol.discard_existing_request()
        num_workers = len(self.datapipes)
        for worker_id, dp in enumerate(self.datapipes):
            worker_info = WorkerInfo(num_workers, worker_id)
            dp.protocol.request_reset_epoch(partial(reset_fn, worker_info=worker_info))

    def request_pause(self):
        # Store results of pending requests
        for idx, dp in enumerate(self.datapipes):
            if dp.protocol.waiting_for_response():
                res = dp.protocol.get_response_next(block=True)
                self.res_buffers[idx].append(res)
        for dp in self.datapipes:
            dp.pause()

    def request_resume(self):
        for dp in self.datapipes:
            dp.resume()


class PrototypeMultiProcessingReadingService(ReadingServiceInterface):
    r"""
    ``PrototypeMultiProcessingReadingService`` that spawns multiple subprocesses to iterate the ``DataPipe`` graph.
    This ``ReadingService`` is still under prototype stage and will replace ``MultiProcessingReadingService`` eventually.

    Args:
        num_workers (int, optional): How many subprocesses to use for data loading.
            ``0`` will be replaced by ``InProcessReadingService`` in the future.
        multiprocessing_context (str, optional): Multiprocessing starting method.
            If method is None then the default context is returned.
            Otherwise, method should be 'fork', 'spawn'.
        worker_prefetch_cnt: (int, 10 by default): Number of data will be prefetched at
            the end of each worker process.
        main_prefetch_cnt: (int, 10 by default): Number of data will be prefetched
            at the end of the whole pipeline in the main process.
        worker_init_fn: (Callable, optional): Function to be called when each worker
            process launches with ``WorkerInfo`` and ``DataPipe``
            as the expected arguments.
        worker_reset_fn: (Callable, optional): Function to be called at the beginning
            of each epoch in each worker process with ``WorkerInfo``
            and ``DataPipe`` as the expected arguments.

    """
    num_workers: int
    multiprocessing_context: Optional[str]
    worker_prefetch_cnt: int
    main_prefetch_cnt: int
    worker_init_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]]
    worker_reset_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]]
    processes: List
    datapipes: List
    end_datapipe: Optional[DataPipe]
    _mp: bool
    _pg: Optional[dist.ProcessGroup]
    _world_size: int
    _rank: int

    def __init__(
        self,
        num_workers: int = 0,
        multiprocessing_context: Optional[str] = None,
        worker_prefetch_cnt: int = 10,
        main_prefetch_cnt: int = 10,
        worker_init_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]] = None,
        worker_reset_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]] = None,
    ) -> None:
        self.num_workers = num_workers
        if multiprocessing_context is not None:
            _all_start_methods = mp.get_all_start_methods()
            assert (
                multiprocessing_context in _all_start_methods
            ), f"Please choose one available multiprocessing context from {_all_start_methods}"
        self.multiprocessing_context = multiprocessing_context
        self.worker_prefetch_cnt = worker_prefetch_cnt
        self.main_prefetch_cnt = main_prefetch_cnt
        self.worker_init_fn = worker_init_fn
        self.worker_reset_fn = worker_reset_fn
        self.processes = []
        self.datapipes = []
        self.end_datapipe = None
        self._mp = num_workers > 0
        self._pg = None
        self._world_size = 1
        self._rank = 0

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        r"""
        ``PrototypeMultiProcessingReadingService`` finds information about sharding,
        separates graph by multiple pieces and reconnects it using queues.
        creates subprocesses.
        """
        if dist.is_available() and dist.is_initialized():
            self._world_size = dist.get_world_size()
            self._rank = dist.get_rank()
            self._pg = dist.new_group(backend="gloo")
            torch.utils.data.graph_settings.apply_sharding(
                datapipe, self._world_size, self._rank, SHARDING_PRIORITIES.DISTRIBUTED
            )
        if not self._mp:
            # TODO(616): Warn and recommend usage of InProcessReadingService
            worker_info = WorkerInfo(1, 0)
            process_init_fn(datapipe, worker_info, self.worker_init_fn)
            self.end_datapipe = datapipe
            return datapipe

        if self.worker_prefetch_cnt > 0:
            datapipe = datapipe.prefetch(self.worker_prefetch_cnt)

        for worker_id in range(self.num_workers):
            worker_info = WorkerInfo(self.num_workers, worker_id)
            call_on_process_init = partial(process_init_fn, worker_info=worker_info, custom_init_fn=self.worker_init_fn)
            ctx = mp.get_context(self.multiprocessing_context)
            # Process contains a ProtocolServer
            (process, req_queue, res_queue) = communication.eventloop.SpawnProcessForDataPipeline(
                ctx,
                datapipe,
                call_on_process_init,
            )
            process.daemon = True
            process.start()
            self.processes.append((process, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
            )
            self.datapipes.append(local_datapipe)

        self.end_datapipe = _IterateQueueDataPipes(self.datapipes)  # type: ignore[assignment]
        if self.main_prefetch_cnt > 0:
            self.end_datapipe = self.end_datapipe.prefetch(self.main_prefetch_cnt)  # type: ignore[union-attr]
        return self.end_datapipe  # type: ignore[return-value]

    def initialize_iteration(self) -> None:
        shared_seed = generate_random_scalar_tensor()
        if self._pg is not None:
            dist.broadcast(shared_seed, src=0, group=self._pg)
        shared_seed_int: int = shared_seed.item()  # type: ignore[assignment]
        _seed_generator = torch.Generator()
        _seed_generator.manual_seed(shared_seed_int)
        torch.utils.data.graph_settings.apply_random_seed(
            self.end_datapipe,  # type: ignore[arg-type]
            _seed_generator,
        )

        assert self.end_datapipe is not None
        if self._mp:
            if self.main_prefetch_cnt > 0:
                # Stop prefetching first
                self.end_datapipe.reset()  # type: ignore[union-attr]
                end_datapipe: DataPipe = self.end_datapipe.source_datapipe
            else:
                end_datapipe = self.end_datapipe
            # Send the shared seed to subprocesses
            dist_info = _DistInfo(shared_seed_int, self._world_size, self._rank)
            call_on_epoch_reset = partial(process_reset_fn, dist_info=dist_info, custom_reset_fn=self.worker_reset_fn)
            end_datapipe.reset_epoch(call_on_epoch_reset)
            end_datapipe.reset()
        # In-process (num_workers == 0)
        else:
            # Technically speaking, we should call `_process_reset_fn` to reset global RNGs
            # for data-related operations. However, it would pollute the state of global RNGs
            # (random, torch and numpy), if users have already seeded them in the main process
            # TODO(ejguan): This should be fixed by adding a method to isolate global RNGs
            pass

    def __del__(self):
        self.finalize()

    def finalize(self) -> None:
        r"""
        ``PrototypeMultiProcessingReadingService`` invalidate states & properly exits all subprocesses.
        """
        # TODO(618): Check if anyone stuck with messages
        def clean_me(process, req_queue, res_queue):
            # TODO(619): Can send terminations simultaneously
            # TODO(620): Make termination a function of QueueWrapperDataPipe (similar to reset)
            req_queue.put(communication.messages.TerminateRequest())
            _ = res_queue.get()
            process.join(default_dl2_worker_join_timeout_in_s)

        for process, req_queue, res_queue in self.processes:
            try:
                clean_me(process, req_queue, res_queue)
            except AttributeError:
                # Due to non-deterministic order of destruction, by the time `finalize` is called,
                # some objects may already be `None`.
                pass
            except TimeoutError:
                pass

        self.processes = []

        if self._pg is not None:
            dist.destroy_process_group(self._pg)
            self._pg = None

    def _pause(self):
        """
        Pauses DataPipes' activities such as prefetching, in order to collect state.
        """
        if self.main_prefetch_cnt > 0 and self.num_workers > 0:
            # Stop prefetching first
            self.end_datapipe.pause()  # type: ignore[union-attr]
            end_datapipe: DataPipe = self.end_datapipe.source_datapipe  # type: ignore[union-attr]
        else:
            end_datapipe = self.end_datapipe  # type: ignore[assignment]
        if self.num_workers > 0:
            end_datapipe.request_pause()
        else:
            raise RuntimeError(
                "If you would like to use `pause` with `PrototypeMultiProcessingReadingService`, "
                "please use more than 0 worker."
            )

    def _resume(self):
        """
        Resumes DataPipes' activities. This is required to be called after `_pause` before
        the DataLoader can keep yielding elements.
        """
        if self.main_prefetch_cnt > 0:
            end_datapipe: DataPipe = self.end_datapipe.source_datapipe  # type: ignore[union-attr]
        else:
            end_datapipe = self.end_datapipe  # type: ignore[assignment]
        if self.num_workers > 0:
            end_datapipe.request_resume()
        else:
            raise RuntimeError(
                "If you would like to use `resume` with `PrototypeMultiProcessingReadingService`, "
                "please use more than 0 worker."
            )
        if self.main_prefetch_cnt > 0 and self.num_workers > 0:
            self.end_datapipe.resume()  # type: ignore[union-attr]


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
    prefetch_factor: Optional[int]
    persistent_workers: bool

    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        multiprocessing_context=None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        if self.num_workers == 0:
            self.prefetch_factor = None
            self.persistent_workers = False
        self.dl_: Optional[DataLoader] = None

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
        return IterableWrapper(self.dl_)  # type: ignore[return-value]

    def finalize(self) -> None:
        if self.persistent_workers and self.dl_ is not None and self.dl_._iterator is not None:
            self.dl_._iterator._shutdown_workers()  # type: ignore[attr-defined]
            self.dl_._iterator = None


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
            datapipe, self._world_size, self._rank, SHARDING_PRIORITIES.DISTRIBUTED
        )
        # Only append FullSyncIterDataPipe if it's not presented at the end of the pipeline
        if not isinstance(datapipe, FullSync):
            datapipe = datapipe.fullsync(self._timeout)
        self._datapipe = datapipe
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
        shared_seed = generate_random_scalar_tensor()
        dist.broadcast(shared_seed, src=0, group=self._pg)
        return shared_seed.item()

    def __del__(self):
        self.finalize()

    def finalize(self) -> None:
        r"""
        Clean up the distributed process group.
        """
        if self._pg is not None:
            dist.destroy_process_group(self._pg)
            self._pg = None
