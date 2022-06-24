# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import torch
from torch.utils.data import DataLoader

from torchdata.dataloader2 import communication
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe


class ReadingServiceInterface(ABC):
    @abstractmethod
    def initialize(self, datapipe: IterDataPipe) -> IterDataPipe:
        """
        ReadingService traverses datapipe graph, finds executable part,
        adapts into its own datapipe, and replaces in datapipe graph.

        Called once in creating DataLoader iterator at first time.

        Args:
            datapipe: IterDataPipe. Original datapipe.

        Return:
            Adapated IterDataPipe.

        Example:
            MultiProcessingReadingService finds information about sharding,
            separates graph by multiple pieces and reconnects it using queues.
            Spawns processes/threads.
        """
        pass

    def finalize(self) -> None:
        """
        ReadingService cleanup states.
        Called in DataLoader shutdown and __del__

        Example:
            MultiProcessingReadingService invalidate states & handle persistent worker.
        """
        pass

    def initialize_iteration(self) -> None:
        """
        ReadingService spin up service.
        Called at the beginning of every time getting DataLoader iterator.

        Example:
            MultiProcessingReadingService starts prefetching items from the graph.
        """
        pass

    def finalize_iteration(self) -> None:
        """
        ReadingService end service.

        Example:
            MultiprocessingReadingService cleans up processes.
        """
        pass


class CheckpointableReadingServiceInterface(ReadingServiceInterface):
    @abstractmethod
    def checkpoint(self) -> bytes:
        """
        ReadingService serialize backend states.
        Called in DataLoader checkpoint.
        """

        pass

    @abstractmethod
    def restore(self, datapipe: IterDataPipe, serialized_state: bytes) -> IterDataPipe:
        """
        ReadingService adapts datapipe and consume serialized state.

        Called once in creating DataLoader iterator at first time.
        Counterpart of `initialize`, which adapt datapipe from scratch.

        Returns:
            Adapted IterDataPipe.
        """
        pass


def _collate_no_op(batch):
    return batch[0]


class _IterateQueueDataPipes:
    def __init__(self, datapipes):
        self.datapipes = datapipes

    def __iter__(self):
        # TODO(VitalyFedyunin): This can traverse pipes in any order, so undeterministic
        not_available = False
        exclude_datapipes: List[Any] = []
        while len(exclude_datapipes) < len(self.datapipes):
            for dp in self.datapipes:
                if dp not in exclude_datapipes:
                    try:
                        value = dp.nonblocking_next()
                        yield value
                    except StopIteration:
                        exclude_datapipes.append(dp)
                    except communication.iter.NotAvailable:
                        not_available = True
            if not_available:
                time.sleep(0.001)


class PrototypeMultiProcessingReadingService(ReadingServiceInterface):
    num_workers: int
    #   pin_memory: bool
    #   timeout: float
    #   worker_init_fn: Optional[Callable[[int], None]]
    #   prefetch_factor: int
    #   persistent_workers: bool

    processes: List
    datapipes: List

    def __init__(
        self,
        num_workers: int = 0,
        multiprocessing_context=None,
    ) -> None:
        self.num_workers = num_workers
        self.multiprocessing_context = multiprocessing_context
        self.processes = []
        self.datapipes = []

    @staticmethod
    def init_datapipe_process(num_workers, worker_id, datapipe):
        # TODO(VitalyFedyunin): Add distributed support
        # TODO(VitalyFedyunin): Add shuffle determinism support
        torch.utils.data.graph_settings.apply_sharding(datapipe, num_workers, worker_id)

    def initialize(self, datapipe: IterDataPipe) -> IterDataPipe:
        if self.num_workers == 0:
            # TODO(VitalyFedyunin): Warn and recommend usage of InPorcessReadingService
            return datapipe
        for worker_id in range(self.num_workers):
            # TODO(VitalyFedyunin): Separate into function, because we also need to apply distributed seed and call it inside process
            call_inside_process = functools.partial(self.init_datapipe_process, self.num_workers, worker_id)
            ctx = mp.get_context(self.multiprocessing_context)
            (process, req_queue, res_queue) = communication.eventloop.SpawnProcessForDataPipeline(
                ctx, datapipe, call_inside_process
            )
            process.start()
            self.processes.append((process, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
            )
            self.datapipes.append(local_datapipe)

        return IterableWrapper(_IterateQueueDataPipes(self.datapipes), deepcopy=False)  # type: ignore[return-value]

    def initialize_iteration(self) -> None:
        for dp in self.datapipes:
            dp.reset_iterator()

    def __del__(self):
        self.finalize()

    def finalize(self) -> None:
        # TODO(VitalyFedyunin): Check if anyone stuck with messages
        def clean_me(process, req_queue, res_queue):
            # TODO(VitalyFedyunin): Can send terminations simulteniously
            # TODO(VitalyFedyunin): Make termination a function of QueueWrapperDataPipe (similar to reset)
            req_queue.put(communication.messages.TerminateRequest())
            _ = res_queue.get()
            process.join()

        for process, req_queue, res_queue in self.processes:
            clean_me(process, req_queue, res_queue)

        self.processes = []


class MultiProcessingReadingService(ReadingServiceInterface):
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

    # Wrap the DataLoader with IterableWrapper to respect type annotation
    def initialize(self, datapipe: IterDataPipe) -> IterDataPipe:
        self.dl_ = DataLoader(
            datapipe,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            multiprocessing_context=self.multiprocessing_context,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            # TODO: `collate_fn` is necessary until we stop using DLv1 https://github.com/pytorch/data/issues/530
            collate_fn=_collate_no_op,
            batch_size=1,  # This reading service assume batching is done via DataPipe
        )
        return IterableWrapper(self.dl_)  # type: ignore[return-value]

    def finalize(self) -> None:
        if self.persistent_workers and self.dl_ is not None and self.dl_._iterator is not None:
            self.dl_._iterator._shutdown_workers()  # type: ignore[attr-defined]
            self.dl_._iterator = None
