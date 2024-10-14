# pyre-unsafe
# import multiprocessing as mp
import queue
import threading
from typing import Callable, List, Literal, TypeVar, Union

import torch

from torch._utils import ExceptionWrapper
from torch.utils.data.sampler import Iterator

from torchdata.nodes import BaseNode, T

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
            if isinstance(item, ExceptionWrapper):
                item.reraise()
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
        mp = torch.multiprocessing
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
            target=_populate_queue, args=(self.source, self.in_q, self._stop, self.sem)
        )

        self._map_threads: List[Union[threading.Thread, mp.Process]] = []
        for worker_id in range(self.num_workers):
            args = (
                worker_id,
                self.in_q,
                self.out_q,
                # self.sem,
                None,
                self.in_order,
                self.udf,
                self._stop if method == "thread" else self._mp_stop,
            )
            self._map_threads.append(
                threading.Thread(target=_apply_udf, args=args)
                if method == "thread"
                else mp.Process(target=_apply_udf, args=args)
            )
        if not self._started:
            self._read_thread.start()
            for t in self._map_threads:
                t.start()
            self._started = True

    def iterator(self) -> Iterator[T]:
        while True:
            if self._stop.is_set():
                self._mp_stop.set()
                break
            try:
                item = self.out_q.get(block=True, timeout=5.0)
            except queue.Empty:
                continue
            self.sem.release()
            if isinstance(item, StopIteration):
                break
            yield item

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        if self._started:
            self._stop.set()
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
