# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time

from collections import deque
from typing import Deque, Optional

from torchdata.dataloader2 import communication

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

PRODUCER_SLEEP_INTERVAL = 0.0001  # Interval between buffer fullfilment checks
CONSUMER_SLEEP_INTERVAL = 0.0001  # Interval between checking items availablitity in buffer


class _PrefetchData:
    def __init__(self, source_datapipe, buffer_size: int):
        self.run_prefetcher = True
        self.prefetch_buffer: Deque = deque()
        self.buffer_size: int = buffer_size
        self.source_datapipe = source_datapipe


@functional_datapipe("prefetch")
class PrefetcherIterDataPipe(IterDataPipe):
    """
    Prefetches elements from the source DataPipe and puts them into a buffer (functional name: ``prefetch``).
    Prefetching performs the operations (e.g. I/O, computations) of the DataPipes up to this one ahead of time
    and stores the result in the buffer, ready to be consume by the subsequent DataPipe. It has no effect aside
    from getting the sample ready ahead of time.

    This is used by ``PrototypeMultiProcessingReadingService`` when the arguments
    ``prefetch_worker`` (for prefetching at each worker process) or
    ``prefetch_mainloop`` (for prefetching at the moain loop) are greater than 0.

    Beyond the built-in use cases, this can be useful to put after I/O DataPipes that have
    expensive I/O operations (e.g. takes a long time to request a file from a remote server).

    Args:
        source_datapipe: IterDataPipe from which samples are prefetched
        buffer_size: the size of the buffer which stores the prefetched samples

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(file_paths).open_files().prefetch(5)
    """

    def __init__(self, source_datapipe, buffer_size: int = 10):
        self.source_datapipe = source_datapipe
        if buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be a positive integer.")
        self.buffer_size = buffer_size
        self.thread: Optional[threading.Thread] = None

    @staticmethod
    def thread_worker(prefetch_data):
        itr = iter(prefetch_data.source_datapipe)
        stop_iteration = False
        while prefetch_data.run_prefetcher:
            if len(prefetch_data.prefetch_buffer) < prefetch_data.buffer_size and not stop_iteration:
                try:
                    item = next(itr)
                    prefetch_data.prefetch_buffer.append(item)
                except StopIteration:
                    stop_iteration = True
                except communication.iter.InvalidStateResetRequired:
                    stop_iteration = True
                except communication.iter.TerminateRequired:
                    prefetch_data.run_prefetcher = False
            elif stop_iteration and len(prefetch_data.prefetch_buffer) == 0:
                prefetch_data.run_prefetcher = False
            else:  # Buffer is full, waiting for main thread to consume items
                # TODO: Calculate sleep interval based on previous consumption speed
                time.sleep(PRODUCER_SLEEP_INTERVAL)

    def __iter__(self):
        if self.buffer_size < 1:
            yield from self.source_datapipe
        else:
            try:
                prefetch_data = _PrefetchData(self.source_datapipe, self.buffer_size)
                self.prefetch_data = prefetch_data
                self.thread = threading.Thread(
                    target=PrefetcherIterDataPipe.thread_worker, args=(prefetch_data,), daemon=True
                )
                self.thread.start()
                while prefetch_data.run_prefetcher:
                    if len(prefetch_data.prefetch_buffer) > 0:
                        yield prefetch_data.prefetch_buffer.popleft()
                    else:
                        # TODO: Calculate sleep interval based on previous availability speed
                        time.sleep(CONSUMER_SLEEP_INTERVAL)
            finally:
                prefetch_data.run_prefetcher = False
                if self.thread is not None:
                    self.thread.join()
                    self.thread = None

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

    def reset(self):
        if self.thread is not None:
            self.prefetch_data.run_prefetcher = False
            self.thread.join()

    def full_stop(self):
        if self.thread is not None:
            self.prefetch_data.run_prefetcher = False
