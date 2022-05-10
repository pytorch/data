# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterator

from torch.utils.data import DataLoader, IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper


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
            Internal:
                DppReadingService finds Dpp executable datapipe,
                gets information (e.g. Koski DF, Conversion Metadata, Post-collate JIT transforms, ...) from it,
                constructs DppIterDataPipe to hold information, and replace original graph with DppIterDataPipe.

                Note: Dpp executable datapipe will always be the 1st DataPipe in graph,
                    which is a TracingArrowDataPipe converted IterDataPipe via collate.
            OSS:
                MultiProcessingReadingService finds information about sharding,
                separates graph by multiple pieces and reconnects it using queues. Spawns processes/threads.
        """
        pass

    def finalize(self) -> None:
        """
        ReadingService cleanup states.
        Called in DataLoader shutdown and __del__

        Example:
            Internal:
                DppReadingService invalidate Dpp Client.

            OSS:
                MultiProcessingReadingService invalidate states & handle persistent worker.
        """
        pass

    def initialize_iteration(self) -> None:
        """
        ReadingService spin up service.
        Called at the beginning of every time getting DataLoader iterator.

        Example:
            Internal:
                DppReadingService spins up DPP service by creating session.

            OSS:
                MultiProcessingReadingService - starts prefetching of the items from the graph.
        """
        pass

    def finalize_iteration(self) -> None:
        """
        ReadingService end service.

        Example:
            Internal:
                DppReadingService destroy DPP service by destroying session.
            OSS:
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

        Example:
            Internal:
                Same as `initialize`.
            OSS:
                Not implemented in H1.
        """
        pass


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
        multiprocessing_context=None,  # pyre-ignore
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context  # pyre-ignore
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
        )
        return IterableWrapper(self.dl_)

    def finalize(self) -> None:
        if (
            self.persistent_workers
            and self.dl_ is not None
            and self.dl_._iterator is not None
        ):
            self.dl_._iterator._shutdown_workers()  # pyre-ignore
            self.dl_._iterator = None  # pyre-ignore
