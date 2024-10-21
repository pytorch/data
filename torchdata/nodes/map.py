# pyre-unsafe
import queue
import threading
from typing import Callable, Iterator, List, Literal, Optional, TypeVar, Union

import torch.multiprocessing as mp

from torchdata.nodes import BaseNode, T

from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper

from ._apply_udf import _apply_udf

from ._populate_queue import _populate_queue


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


class ParallelMapper(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[X],
        map_fn: Callable[[X], T],
        num_workers: int,
        in_order: bool = True,
        method: Literal["thread", "process"] = "thread",
    ):
        self.source = source
        self.udf = map_fn
        self.num_workers = num_workers
        self.in_order = in_order

        self.in_q: Union[queue.Queue, mp.Queue] = queue.Queue() if method == "thread" else mp.Queue()
        self.out_q: Union[queue.Queue, mp.Queue] = queue.Queue() if method == "thread" else mp.Queue()
        self.sem = threading.BoundedSemaphore(value=2 * num_workers)

        self._started = False
        self._stop = threading.Event()
        self._mp_stop = mp.Event()
        self._read_thread = threading.Thread(
            target=_populate_queue, args=(self.source, self.in_q, self.sem, self._stop)
        )
        self._method = method

        self._map_threads: List[Union[threading.Thread, mp.Process]] = []

    def iterator(self) -> Iterator[T]:
        if not self._started:
            for worker_id in range(self.num_workers):
                args = (
                    worker_id,
                    self.in_q,
                    self.out_q,
                    self.in_order,
                    self.udf,
                    self._stop if self._method == "thread" else self._mp_stop,
                )
                self._map_threads.append(
                    threading.Thread(target=_apply_udf, args=args)
                    if self._method == "thread"
                    else mp.Process(target=_apply_udf, args=args)
                )
            self._read_thread.start()
            for t in self._map_threads:
                t.start()
            self._started = True

        exception: Optional[ExceptionWrapper] = None
        while True:
            if self._stop.is_set():
                yield from self._flush_queues()
                self._mp_stop.set()
                break
            try:
                item = self.out_q.get(block=True, timeout=1.0)
                self.sem.release()
            except queue.Empty:
                continue
            if isinstance(item, StopIteration):
                yield from self._flush_queues()
                break
            elif isinstance(item, ExceptionWrapper):
                exception = item
                break
            yield item

        self._stop.set()
        self._mp_stop.set()
        if exception is not None:
            exception.reraise()
        self._shutdown()

    def _flush_queues(self):
        while self.sem._value < 2 * self.num_workers:
            x = self.out_q.get(block=True, timeout=5.0)
            self.sem.release()
            yield x

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        if self._started:
            self._stop.set()
            self._mp_stop.set()
            for _ in range(5):
                self._read_thread.join(timeout=0.1)
                if not self._read_thread.is_alive():
                    break
            for t in self._map_threads:
                for _ in range(5):
                    t.join(timeout=0.1)
                    if not t.is_alive():
                        break

            self._started = False


_WorkerType = Callable[[BaseNode, queue.Queue, threading.BoundedSemaphore, threading.Event], None]


class _SingleThreadedMapper(Iterator[T]):
    """Utility Iterator for performing mapping with a single thread.
    Because only a single thread is used, we don't need an input queue to guard
    against multiple threads reading from the same iterator. This is used for
    Prefetcher and PinMemory.
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
