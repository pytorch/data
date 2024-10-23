# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
import threading
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Protocol, TypeVar, Union

import torch.multiprocessing as mp
from torch import TYPE_CHECKING
from torchdata.nodes import BaseNode, T
from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper

from ._apply_udf import _apply_udf

from ._populate_queue import _populate_queue

if TYPE_CHECKING:

    class _MultiprocessContext(Protocol):
        def Process(self, *args, **kwargs):
            ...

        def Event(self, *args, **kwargs):
            ...

        def Queue(self, *args, **kwargs):
            ...


X = TypeVar("X")


class Mapper(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[X],
        map_fn: Callable[[X], T],
    ):
        self.source = source
        self.map_fn = map_fn

    def iterator(self) -> Iterator[T]:
        for item in self.source:
            yield self.map_fn(item)


def _sort_worker(in_q: Union[queue.Queue, mp.Queue], out_q: queue.Queue, stop_event: threading.Event):
    buffer: Dict[int, Any] = {}
    cur_idx = 0
    while not stop_event.is_set():
        try:
            x, idx = in_q.get(block=True, timeout=0.1)
        except queue.Empty:
            continue
        if idx == cur_idx:
            out_q.put((x, cur_idx), block=False)
            cur_idx += 1
        else:
            if idx in buffer:
                # This is the easiest way to create an exception wrapper
                try:
                    raise ValueError(f"Duplicate index {idx=}, {buffer.keys()=}, {x=}")
                except Exception:
                    x = ExceptionWrapper(where="in _sort_worker")
                out_q.put((x, idx), block=False)
                break
            buffer[idx] = x
        while cur_idx in buffer:
            out_q.put((buffer.pop(cur_idx), cur_idx), block=False)
            cur_idx += 1


class _ParallelMapperIter(Iterator[T]):
    def __init__(
        self,
        source: BaseNode[X],
        map_fn: Callable[[X], T],
        num_workers: int,
        in_order: bool,
        method: Literal["thread", "process"],
        mp_context: _MultiprocessContext,
    ):
        self.source = source
        self.map_fn = map_fn
        self.num_workers = num_workers
        self.in_order = in_order
        self.method = method
        self.mp_context = mp_context

        self._in_q: Union[queue.Queue, mp.Queue] = queue.Queue() if method == "thread" else mp_context.Queue()
        self._intermed_q: Union[queue.Queue, mp.Queue] = queue.Queue() if method == "thread" else mp_context.Queue()
        self._max_tasks = 2 * self.num_workers
        self._sem = threading.BoundedSemaphore(value=self._max_tasks)

        self._done = False

        self._stop = threading.Event()
        self._mp_stop = mp_context.Event()

        self._read_thread = threading.Thread(
            target=_populate_queue,
            args=(self.source, self._in_q, self._sem, self._stop, True),
        )
        self._map_threads: List[Union[threading.Thread, mp.Process]] = []
        for worker_id in range(self.num_workers):
            args = (
                worker_id,
                self._in_q,
                self._intermed_q,
                self.map_fn,
                self._stop if self.method == "thread" else self._mp_stop,
            )
            self._map_threads.append(
                threading.Thread(target=_apply_udf, args=args)
                if self.method == "thread"
                else mp_context.Process(target=_apply_udf, args=args)
            )
        self._sort_q: queue.Queue = queue.Queue()
        self._sort_thread = threading.Thread(target=_sort_worker, args=(self._intermed_q, self._sort_q, self._stop))

        self._out_q = self._intermed_q
        if self.in_order:
            self._out_q = self._sort_q

        self._read_thread.start()
        for t in self._map_threads:
            t.start()
        if self.in_order:
            self._sort_thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._stop.is_set():
                raise StopIteration()
            elif self._done and self._sem._value == self._max_tasks:
                # Don't stop if we still have items in the queue
                self._stop.set()
                self._mp_stop.set()
                raise StopIteration()
            try:
                item, idx = self._out_q.get(block=True, timeout=1.0)
            except queue.Empty:
                continue

            if isinstance(item, StopIteration):
                self._done = True
                self._sem.release()
                # Make sure queues are flushed before returning early
                continue
            elif isinstance(item, ExceptionWrapper):
                if not isinstance(item, StartupExceptionWrapper):
                    self._sem.release()
                item.reraise()
            else:
                self._sem.release()
                return item

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        self._stop.set()
        self._mp_stop.set()
        if self._read_thread.is_alive():
            self._read_thread.join(timeout=0.5)
        if self._sort_thread.is_alive():
            self._sort_thread.join(timeout=0.5)
        for t in self._map_threads:
            if t.is_alive():
                t.join(timeout=0.5)


class ParallelMapper(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[X],
        map_fn: Callable[[X], T],
        num_workers: int,
        in_order: bool = True,
        method: Literal["thread", "process"] = "thread",
        multiprocessing_context: Optional[str] = "forkserver",
    ):
        assert method in ["thread", "process"]
        self.source = source
        self.map_fn = map_fn
        self.num_workers = num_workers
        self.in_order = in_order
        self.method = method
        self.multiprocessing_context = multiprocessing_context
        self._mp_context: Any = mp
        if self.method == "process" and self.multiprocessing_context is not None:
            self._mp_context = mp.get_context(self.multiprocessing_context)

    def iterator(self) -> Iterator[T]:
        return _ParallelMapperIter(
            source=self.source,
            map_fn=self.map_fn,
            num_workers=self.num_workers,
            in_order=self.in_order,
            method=self.method,
            mp_context=self._mp_context,
        )


_WorkerType = Callable[[BaseNode, queue.Queue, threading.BoundedSemaphore, threading.Event], None]


class _SingleThreadedMapper(Iterator[T]):
    """Utility Iterator for performing mapping with a single thread.
    Because only a single thread is used, we don't need an input queue to guard
    against multiple threads reading from the same iterator. This is used for
    Prefetcher and PinMemory.

    A thread is started on __init__ and stopped on __del__/_shutdown.
    The thread runs _populate_queue, which acquires a BoundedSemaphore with initial value
    of `prefetch_factor`.

    When next() is called on this iterator, it will block until an item is available on _q.
    Next will perform the following depending on what is pulled from the q:
    - StopIteration: raise StopIteration. Any subsequent next() calls will also raise StopIteration
    - ExceptionWrapper: call reraise() on the exception wraper
    - any other item: return the item

    A Bounded semaphore is used to limit concurrency and memory utilization.
    If N items have been pulled from the source, and M items have been yielded by this iterator,
    we maintain the invariant that semaphore.value + (N - M) == prefetch_factor (modulo
    non-atomicness of operations).

    _populate_queue calls semaphore.acquire. When we pull an item from the queue, we
    call semaphore.release (unless it's a StartupExceptionWrapper, because _populate_queue
    does not acquire sempahores in this case). All outstanding items are either being
    processed in _populate_queue, in the _q, or about to be returned by an in-flight next() call.
    """

    def __init__(self, source: BaseNode[T], prefetch_factor: int, worker: _WorkerType):
        self.source = source
        self.prefetch_factor = prefetch_factor
        self.worker = worker

        self._q: queue.Queue = queue.Queue()
        self._sem = threading.BoundedSemaphore(value=prefetch_factor)
        self._stop_event = threading.Event()

        self._thread = threading.Thread(
            target=self.worker,
            args=(self.source, self._q, self._sem, self._stop_event),
        )
        self._thread.start()
        self._stopped = False

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self):
        if self._stopped:
            raise StopIteration()

        while True:
            try:
                item = self._q.get(block=True, timeout=0.1)
                break
            except queue.Empty:
                continue

        if isinstance(item, StopIteration):
            self._stopped = True
            self._sem.release()
            self._stop_event.set()
            raise item
        elif isinstance(item, ExceptionWrapper):
            self._stopped = True
            if not isinstance(item, StartupExceptionWrapper):
                # We don't need to release for startup exceptions
                self._sem.release()
            self._stop_event.set()
            item.reraise()
        else:
            self._sem.release()
        return item

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        self._stop_event.set()
        self._thread.join(timeout=0.1)
