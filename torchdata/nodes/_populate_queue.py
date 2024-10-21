# pyre-unsafe
import queue
import threading
from typing import Iterable

from torch._utils import ExceptionWrapper


def _populate_queue(
    source: Iterable,
    q: queue.Queue,
    stop_event: threading.Event,
    semaphore: threading.BoundedSemaphore,
):
    """Note that this is only intended to be used
    by a single thread at once. Each instance creates its own iter for source so
    if this is called with multiple threads, you may get duplicates if
    source is not sharded properly.
    """
    try:
        src_iter = iter(source)
    except Exception:
        e = ExceptionWrapper(where="in _populate_queue startup for device")
        q.put(e)
        return

    while not stop_event.is_set():
        if not semaphore.acquire(blocking=True, timeout=1.0):
            continue
        try:
            x = next(src_iter)  # FIXME: This may hang!
        except StopIteration as e:
            q.put(e)
            break
        except Exception:
            x = ExceptionWrapper(where="in _populate_queue")
        try:
            q.put(x, block=False)  # Semaphore should prevent this from throwing
        except queue.Full:
            raise RuntimeError("Queue should not be full")
