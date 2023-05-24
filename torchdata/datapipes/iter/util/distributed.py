# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading

import time

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from functools import partial
from typing import Callable, Deque, final, Iterator, Optional, TypeVar

import torch
import torch.distributed as dist

from torchdata._constants import default_timeout_in_s
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.iter.util.prefetcher import PRODUCER_SLEEP_INTERVAL


T_co = TypeVar("T_co", covariant=True)


__all__ = ["Expected", "FullSyncIterDataPipe", "PrefetchTimeoutError"]


class PrefetchTimeoutError(RuntimeError):
    def __init__(self, timeout: int) -> None:
        super().__init__(f"Fail to fetch data within {timeout} seconds")
        self.timeout = timeout


class _EndOfPrefetch:
    ...


@dataclass
class Expected:
    r"""
    Expected data provided to callback function in ``_PrefetchExecutor``.
    """
    index: int
    error: Optional[BaseException] = None

    def has_error(self) -> bool:
        return self.error is not None


class _PrefetchExecutor:
    # TODO: Improvement - merge with the `_PrefetchData` class of prefetcher.py
    #       May not be possible right now due to circular import
    def __init__(
        self,
        datapipe_iterator: Iterator,
        prefetch_size: int = 1,
        callback_fn: Optional[Callable[[Expected], None]] = None,
        timeout: int = default_timeout_in_s,
    ) -> None:
        self.datapipe_iterator = datapipe_iterator
        self.prefetch_size = prefetch_size
        self.callback_fn = callback_fn
        self.timeout = timeout
        # Use max_workers as 1 to guarantee the order of data fetched from iterator
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._futures: Deque[Future] = deque()
        self._lock = threading.RLock()
        # `_end_flag` indicates the end of epoch or an exception has been raised,
        # with the exception being handled by `callback_fn`
        self._end_flag: bool = False
        self._paused: bool = False
        self._is_shutdown: bool = False  # indicates if `_executor` has been shutdown by `shutdown` method
        self._idx = 0
        for _ in range(prefetch_size):
            with self._lock:
                if self._end_flag:
                    break
            fetch_future: Future = self._executor.submit(self.fetch_next)
            fetch_future.add_done_callback(partial(self._done_callback_fn, self._idx))
            self._futures.append(fetch_future)
            with self._lock:
                self._idx += 1

    def fetch_next(self):
        while self._paused:
            time.sleep(PRODUCER_SLEEP_INTERVAL * 10)
        return next(self.datapipe_iterator)

    def _done_callback_fn(self, index: int, f: Future):
        if f.exception():
            with self._lock:
                self._end_flag = True
        if self.callback_fn is not None:
            # Invoke `callback_fn` in order to set `FullSyncDP._done_callback` to `True`
            self.callback_fn(Expected(index, f.exception()))

    def return_next(self):
        if self._futures:
            fetch_future = self._futures.popleft()
            try:
                data = fetch_future.result(timeout=self.timeout)
            except TimeoutError:
                raise PrefetchTimeoutError(self.timeout)
            with self._lock:
                if not self._end_flag and not self._is_shutdown:
                    next_future = self._executor.submit(self.fetch_next)
                    next_future.add_done_callback(partial(self._done_callback_fn, self._idx))
                    self._futures.append(next_future)
                    self._idx += 1
        else:
            data = _EndOfPrefetch()
        return data

    def shutdown(self):
        self._paused = False
        self._is_shutdown = True
        while self._futures:
            self._futures.popleft().cancel()
        self._executor.shutdown(wait=True)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False


@functional_datapipe("fullsync")
class FullSyncIterDataPipe(IterDataPipe[T_co]):
    r"""
    Synchronizes data across distributed processes to prevent hanging during training,
    which is caused by uneven sharded data (functional name: ``fullsync``). It stops
    when the shortest distributed shard is exhausted. It would be appended at the end
    of the graph of ``DataPipe`` by ``DistributedReadingService`` automatically.

    Args:
        datapipe: IterDataPipe that needs to be synchronized
        timeout: Timeout for prefetching data in seconds. Default value equals to 30 minutes

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> # Distributed training with world size 2
        >>> world_size = 2
        >>> dp = IterableWrapper(list(range(23))).sharding_filter()
        >>> torch.utils.data.graph_settings.apply_sharding(dp, world_size, rank)
        >>> # Rank 0 has 12 elements; Rank 1 has 11 elements
        >>> for d in dp:
        ...     model(d)  # Hanging at the end of epoch due to uneven sharding
        >>> dp = dp.fullsync()
        >>> # Both ranks have 11 elements
        >>> for d in dp:
        ...     model(d)  # Not hanging anymore
    """

    def __init__(self, datapipe: IterDataPipe, timeout=default_timeout_in_s):
        if not dist.is_available():
            raise RuntimeError("Torch Distributed is required to be available")
        self.datapipe = datapipe
        self.timeout: int = timeout

        self._process_group: Optional[dist.ProcessGroup] = None
        self._world_size: int = 1

        self._lock = threading.RLock()
        self._cv = threading.Condition(lock=self._lock)
        self._executor: Optional[_PrefetchExecutor] = None
        # Use single values rather than deques for the following variables
        # because fullsync only prefetches 1 element
        self._error = None
        self._sync_counter = torch.tensor([0], dtype=torch.int32)
        self._done_callback = False

    def _callback_fn(self, exp: Expected) -> None:
        with self._cv:
            if exp.has_error():
                if not isinstance(exp.error, StopIteration):
                    self._error = exp.error  # type: ignore[assignment]
                self._sync_counter = torch.tensor([0], dtype=torch.int32)
            else:
                self._sync_counter = torch.tensor([1], dtype=torch.int32)
            dist.all_reduce(
                tensor=self._sync_counter,
                op=dist.ReduceOp.SUM,
                group=self._process_group,
            )
            self._done_callback = True
            self._cv.notify()

    def __iter__(self) -> Iterator[T_co]:
        assert self._executor is None
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Torch Distributed is required to be initialized to use `FullSync`.")

        if self._process_group is None:
            self._process_group = dist.new_group(backend="gloo")
        self._world_size = dist.get_world_size()

        if self._world_size == 1:  # The below functionalities are not needed if `_world_size == 1`
            yield from self.datapipe
            return

        self._executor = _PrefetchExecutor(iter(self.datapipe), 1, self._callback_fn, self.timeout)
        while True:
            with self._cv:
                is_success = self._cv.wait_for(
                    lambda: self._done_callback is True,
                    self.timeout,
                )
                if not is_success:
                    raise PrefetchTimeoutError(self.timeout)
                if self._error is not None:
                    raise self._error
                if bool(self._sync_counter < self._world_size):
                    break
                self._done_callback = False
                data = self._executor.return_next()  # type: ignore[attr-defined]
            if isinstance(data, _EndOfPrefetch):
                break
            yield data

    @final
    def reset(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        self._world_size = 1
        with self._cv:
            self._error = None
            self._sync_counter = torch.tensor([0], dtype=torch.int32)
            self._done_callback = False

    def is_replicable(self):
        return False

    def __getstate__(self):
        state = (
            self.datapipe,
            self.timeout,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        self.datapipe, self.timeout = state
        self._process_group = None
        self._world_size = 1
        self._lock = threading.RLock()
        self._cv = threading.Condition(lock=self._lock)
        self._executor = None
        self._error = None
        self._sync_counter = torch.tensor([0], dtype=torch.int32)
        self._done_callback = False

    @final
    def pause(self):
        if self._executor is not None:
            self._executor.pause()

    @final
    def resume(self):
        if self._executor is not None:
            self._executor.resume()

    @final
    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __del__(self):
        self.shutdown()
