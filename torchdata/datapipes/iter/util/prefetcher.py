# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time

from collections import deque
from typing import Deque, final, Optional, Sized

import torch

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import pin_memory_fn

PRODUCER_SLEEP_INTERVAL = 0.0001  # Interval between buffer fulfillment checks
CONSUMER_SLEEP_INTERVAL = 0.0001  # Interval between checking items availability in buffer


class _PrefetchData:
    def __init__(self, source_datapipe, buffer_size: int, profiling: bool = False):
        self.run_prefetcher: bool = True
        self.prefetch_buffer: Deque = deque()
        self.buffer_size: int = buffer_size
        self.source_datapipe = source_datapipe
        self.stop_iteration: bool = False
        self.paused: bool = False

        self.profiling = profiling
        self.sleep_time = 0
        self.fetch_count = 0
        self.aggregate_queue_size_at_yield = 0
        self.initial_time = None
        self.latest_time = None

    def get_average_fetch_time(self) -> float:
        if not self.profiling:
            raise RuntimeError("Prefetcher must be profiling to know the average_fetch_time.")
        total_time = self.latest_time - self.initial_time
        return total_time / self.fetch_count

    def get_percent_sleep_time(self) -> float:
        if not self.profiling:
            raise RuntimeError("Prefetcher must be profiling to know the percent_sleep_time.")
        total_time = self.latest_time - self.initial_time
        return self.sleep_time / total_time

    def get_average_queue_size_at_yield(self, n_samples_yielded: int) -> float:
        if not self.profiling:
            raise RuntimeError("Prefetcher must be profiling to know the average_queue_size_at_yield.")
        return (self.aggregate_queue_size_at_yield / n_samples_yielded) / self.buffer_size


@functional_datapipe("prefetch")
class PrefetcherIterDataPipe(IterDataPipe):
    r"""
    Prefetches elements from the source DataPipe and puts them into a buffer (functional name: ``prefetch``).
    Prefetching performs the operations (e.g. I/O, computations) of the DataPipes up to this one ahead of time
    and stores the result in the buffer, ready to be consumed by the subsequent DataPipe. It has no effect aside
    from getting the sample ready ahead of time.

    This is used by ``MultiProcessingReadingService`` when the arguments
    ``worker_prefetch_cnt`` (for prefetching at each worker process) or
    ``main_prefetch_cnt`` (for prefetching at the main loop) are greater than 0.

    Beyond the built-in use cases, this can be useful to put after I/O DataPipes that have
    expensive I/O operations (e.g. takes a long time to request a file from a remote server).

    Args:
        source_datapipe: IterDataPipe from which samples are prefetched
        buffer_size: the size of the buffer which stores the prefetched samples

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(file_paths).open_files().prefetch(5)
    """

    def __init__(self, source_datapipe, buffer_size: int = 10, profiling: bool = False):
        self.source_datapipe = source_datapipe
        if buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be a positive integer.")
        self.buffer_size = buffer_size
        self.thread: Optional[threading.Thread] = None
        self.profiling = profiling

    @staticmethod
    def thread_worker(prefetch_data: _PrefetchData):
        itr = iter(prefetch_data.source_datapipe)
        while not prefetch_data.stop_iteration:
            # Run if not paused
            while prefetch_data.run_prefetcher:
                if len(prefetch_data.prefetch_buffer) < prefetch_data.buffer_size:
                    try:
                        item = next(itr)
                        if prefetch_data.profiling:
                            t = time.time()
                            if prefetch_data.fetch_count == 0:
                                prefetch_data.initial_time = t
                            prefetch_data.latest_time = t
                            prefetch_data.fetch_count += 1
                        prefetch_data.prefetch_buffer.append(item)
                    except Exception as e:
                        prefetch_data.run_prefetcher = False
                        prefetch_data.stop_iteration = True
                        prefetch_data.prefetch_buffer.append(e)
                else:  # Buffer is full, waiting for main thread to consume items
                    # TODO: Calculate sleep interval based on previous consumption speed
                    if prefetch_data.profiling:
                        prefetch_data.sleep_time += PRODUCER_SLEEP_INTERVAL
                    time.sleep(PRODUCER_SLEEP_INTERVAL)
            prefetch_data.paused = True
            # Sleep longer when this prefetcher thread is paused
            time.sleep(PRODUCER_SLEEP_INTERVAL * 10)

    def get_average_fetch_time(self):
        if self.prefetch_data and self.profiling:
            return self.prefetch_data.get_average_fetch_time()
        else:
            raise RuntimeError("Usage of prefetch worker and `profiling=True` are required to get this statistic.")

    def get_percent_sleep_time(self):
        if self.prefetch_data and self.profiling:
            return self.prefetch_data.get_percent_sleep_time()
        else:
            raise RuntimeError("Usage of prefetch worker and `profiling=True` are required to get this statistic.")

    def get_average_queue_size_at_yield(self):
        """If this percentage is very low, it may mean consumers are faster than what prefetch allows."""
        if self.prefetch_data and self.profiling:
            return self.prefetch_data.get_average_queue_size_at_yield(self._number_of_samples_yielded)
        else:
            raise RuntimeError("Usage of prefetch worker and `profiling=True` are required to get this statistic.")

    def __iter__(self):
        try:
            prefetch_data = _PrefetchData(self.source_datapipe, self.buffer_size, profiling=self.profiling)
            self.prefetch_data = prefetch_data
            thread = threading.Thread(target=PrefetcherIterDataPipe.thread_worker, args=(prefetch_data,), daemon=True)
            thread.start()
            self.thread = thread

            # Lazily import to prevent circular import
            from torchdata.dataloader2 import communication

            while not prefetch_data.stop_iteration or len(prefetch_data.prefetch_buffer) > 0:
                if len(prefetch_data.prefetch_buffer) > 0:
                    data = prefetch_data.prefetch_buffer.popleft()
                    if isinstance(data, Exception):
                        if isinstance(data, (StopIteration, communication.iter.TerminateRequired)):
                            break
                        raise data
                    if self.profiling:
                        prefetch_data.aggregate_queue_size_at_yield += len(prefetch_data.prefetch_buffer)
                    yield data
                else:
                    time.sleep(CONSUMER_SLEEP_INTERVAL)
        finally:
            prefetch_data.run_prefetcher = False
            prefetch_data.stop_iteration = True
            thread.join()

    def __getstate__(self):
        """
        Getting state in threading environment requires next operations:
            1) Stopping of the producer thread.
            2) Saving buffer.
            3) Adding lazy restart of producer thread when __next__ is called again
              (this will guarantee that you only change state of the source_datapipe
               after entire state of the graph is saved).
        """
        # TODO: Update __getstate__ and __setstate__ to support snapshotting and restoration
        return {"source_datapipe": self.source_datapipe, "buffer_size": self.buffer_size}

    def __setstate__(self, state):
        self.source_datapipe = state["source_datapipe"]
        self.buffer_size = state["buffer_size"]
        self.thread = None

    @final
    def reset(self):
        if self.thread is not None:
            self.prefetch_data.run_prefetcher = False
            self.prefetch_data.stop_iteration = True
            self.prefetch_data.paused = False
            self.thread.join()
            self.thread = None

    def pause(self):
        if self.thread is not None:
            assert self.prefetch_data is not None
            self.prefetch_data.run_prefetcher = False
            if self.thread.is_alive():
                # Blocking until the thread is paused
                while not self.prefetch_data.paused:
                    time.sleep(PRODUCER_SLEEP_INTERVAL * 10)

    @final
    def resume(self):
        if self.thread is not None and (
            not self.prefetch_data.stop_iteration or len(self.prefetch_data.prefetch_buffer) > 0
        ):
            assert self.prefetch_data is not None
            self.prefetch_data.run_prefetcher = True
            self.prefetch_data.paused = False

    def __len__(self) -> int:
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe("pin_memory")
class PinMemoryIterDataPipe(PrefetcherIterDataPipe):
    r"""
    Prefetches one element from the source DataPipe and moves it to pinned memory (functional name: ``pin_memory``).
    When used with ``MultiProcessingReadingService``, this DataPipe would be kept in the main process to prevent
    duplicated CUDA context creation.

    Args:
        source_datapipe: IterDataPipe from which samples are moved to pinned memory.
        device: The device to pin samples.
        pin_memory_fn: Optional callable function to move data to pinned memory.
            A ``pin_memory_fn`` to handle general objects is provided by default.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(file_paths).open_files().readlines().map(tokenize_fn).pin_memory()
    """

    def __init__(self, source_datapipe, device=None, pin_memory_fn=pin_memory_fn):
        if not torch.cuda.is_available():
            raise RuntimeError("``pin_memory`` can only be used when CUDA is available.")
        # TODO: Add support for dynamic buffer based on the available size of pinned memory
        super().__init__(source_datapipe, buffer_size=2)
        if device is None:
            device = torch.cuda.current_device()
        self.device = device
        self.pin_memory_fn = pin_memory_fn

    def is_replicable(self) -> bool:
        return False

    @staticmethod
    def thread_worker(prefetch_data: _PrefetchData, pin_memory_fn, device):  # type: ignore[override]
        itr = iter(prefetch_data.source_datapipe)
        while not prefetch_data.stop_iteration:
            # Run if not paused
            while prefetch_data.run_prefetcher:
                if len(prefetch_data.prefetch_buffer) < prefetch_data.buffer_size:
                    try:
                        item = pin_memory_fn(next(itr), device)
                        prefetch_data.prefetch_buffer.append(item)
                    except Exception as e:
                        prefetch_data.run_prefetcher = False
                        prefetch_data.stop_iteration = True
                        prefetch_data.prefetch_buffer.append(e)
                else:  # Buffer is full, waiting for main thread to consume items
                    # TODO: Calculate sleep interval based on previous consumption speed
                    time.sleep(PRODUCER_SLEEP_INTERVAL)
            # Sleep longer when this prefetcher thread is paused
            time.sleep(PRODUCER_SLEEP_INTERVAL * 10)

    def __iter__(self):
        try:
            prefetch_data = _PrefetchData(self.source_datapipe, self.buffer_size)
            self.prefetch_data = prefetch_data
            thread = threading.Thread(
                target=PinMemoryIterDataPipe.thread_worker,
                args=(prefetch_data, self.pin_memory_fn, self.device),
                daemon=True,
            )
            thread.start()
            self.thread = thread

            # Lazily import to prevent circular import
            from torchdata.dataloader2 import communication

            while not prefetch_data.stop_iteration or len(prefetch_data.prefetch_buffer) > 0:
                if len(prefetch_data.prefetch_buffer) > 0:
                    data = prefetch_data.prefetch_buffer.popleft()
                    if isinstance(data, Exception):
                        if isinstance(data, (StopIteration, communication.iter.TerminateRequired)):
                            break
                        raise data
                    yield data
                else:
                    time.sleep(CONSUMER_SLEEP_INTERVAL)
        finally:
            prefetch_data.run_prefetcher = False
            prefetch_data.stop_iteration = True
            thread.join()

    def __getstate__(self):
        state = super().__getstate__()
        state["pin_memory_fn"] = self.pin_memory_fn
        state["device"] = self.device
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.pin_memory_fn = state["pin_memory_fn"]
        self.device = state["device"]
