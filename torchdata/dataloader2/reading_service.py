# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Callable, Optional

from torch.utils.data import DataLoader
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
        )
        return IterableWrapper(self.dl_)  # type: ignore[return-value]

    def finalize(self) -> None:
        if self.persistent_workers and self.dl_ is not None and self.dl_._iterator is not None:
            self.dl_._iterator._shutdown_workers()  # type: ignore[attr-defined]
            self.dl_._iterator = None
