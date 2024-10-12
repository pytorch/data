# pyre-unsafe
import queue
import threading
from typing import Iterator, Optional

from torch._utils import ExceptionWrapper

from torch.utils.data import IterableDataset
from torchdata.nodes import BaseNode

from ._populate_queue import _populate_queue


class Prefetcher[T](BaseNode[T]):
    def __init__(self, source: IterableDataset, prefetch_factor: int):
        self.source = source
        self.prefetch_factor = prefetch_factor
        self.q = queue.Queue(maxsize=prefetch_factor)
        self.sem = threading.BoundedSemaphore(value=prefetch_factor)
        self._stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def iterator(self) -> Iterator[T]:
        if self.thread is None:
            self._stop_event.clear()
            self.thread = threading.Thread(
                target=_populate_queue,
                args=(self.source, self.q, self._stop_event, self.sem),
            )
            self.thread.start()
        while True:
            try:
                value = self.q.get(block=True, timeout=0.1)
                self.sem.release()
                if isinstance(value, StopIteration):
                    break
                yield value
                if isinstance(value, ExceptionWrapper):
                    self._stop_event.set()
                    break
            except queue.Empty:
                continue
        self._shutdown()

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        if self.thread is not None:
            self._stop_event.set()
            self.thread.join(timeout=0.1)
            self.thread = None
