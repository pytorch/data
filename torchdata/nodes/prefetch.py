# pyre-unsafe
import queue
import threading
from typing import Iterator, Optional

from torch._utils import ExceptionWrapper

from torchdata.nodes import BaseNode, T

from ._populate_queue import _populate_queue


class Prefetcher(BaseNode[T]):
    def __init__(self, source: BaseNode[T], prefetch_factor: int):
        self.source = source
        self.prefetch_factor = prefetch_factor
        self.q: queue.Queue = queue.Queue(maxsize=prefetch_factor)
        self.sem = threading.BoundedSemaphore(value=prefetch_factor)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def iterator(self) -> Iterator[T]:
        if not self._started:
            self._started = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=_populate_queue,
                args=(self.source, self.q, self._stop_event, self.sem),
            )
            self._thread.start()

        exception: Optional[ExceptionWrapper] = None
        while True:
            try:
                item = self.q.get(block=True, timeout=0.1)
            except queue.Empty:
                continue
            self.sem.release()
            if isinstance(item, StopIteration):
                break
            elif isinstance(item, ExceptionWrapper):
                exception = item
                break
            yield item
        self._stop_event.set()
        if exception is not None:
            exception.reraise()
        self._shutdown()

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)
            self._thread = None
        self._started = False
