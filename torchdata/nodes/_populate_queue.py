# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
import threading
from dataclasses import dataclass
from typing import Iterable

from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper

from .constants import QUEUE_TIMEOUT


@dataclass
class _MonotonicIndex:
    initial: int = 0

    def __post_init__(self):
        self._idx = self.initial

    def get(self) -> int:
        idx = self._idx
        self._idx += 1
        return idx


def _populate_queue(
    source: Iterable,
    q: queue.Queue,
    semaphore: threading.BoundedSemaphore,
    stop_event: threading.Event,
    add_index: bool = False,
):
    """_populate_queue calls `iter(source)` to get an iterator `it`, waits for semaphore.acquire,
    and puts its outputs onto q. It never releases the sempahore. It continues to put items on the
    q as long as it can acquire the sempahore, stop_event is not set, and StopIteration has not
    been thrown by the `it`.

    If add_index = True, this function will always put tuples of (x, idx) on the q where idx
    starts from 0 and is monotonically increasing. x may be the output of next(it), StopIteration,
    or an ExceptionWrapper.

    If there is an exception raised during the call to `iter(source)`, this function does not
    wait to acquire sempahore before putting StartupExceptionWrapper on q.

    Note: this is only intended to be used by a single thread at once. Each instance
    creates its own iter for source so if this is called with multiple threads, you may get
    duplicates if source is not sharded properly.
    """

    # Include a monotonic index starting from 0 to each item in the queue
    idx = _MonotonicIndex()

    def _put(item, block: bool = True):
        if add_index:
            q.put((item, idx.get()), block=block)
        else:
            q.put(item, block=block)

    try:
        src_iter = iter(source)
    except Exception:
        e = StartupExceptionWrapper(where="in _populate_queue startup for device")
        _put(e)
        return

    while not stop_event.is_set():
        if not semaphore.acquire(blocking=True, timeout=QUEUE_TIMEOUT):
            continue
        try:
            item = next(src_iter)  # FIXME: This may hang!
        except StopIteration as e:
            _put(e)
            break
        except Exception:
            item = ExceptionWrapper(where="in _populate_queue")
        try:
            _put(item, block=False)  # Semaphore should prevent this from throwing
        except queue.Full:
            raise RuntimeError("Queue should not be full")
