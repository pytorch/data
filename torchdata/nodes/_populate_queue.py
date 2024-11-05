# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch.multiprocessing as mp

from torchdata.nodes.base_node import BaseNode

from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper
from torchdata.nodes.snapshot_store import SnapshotStore

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
    source: BaseNode,
    q: Union[queue.Queue, mp.Queue],
    snapshot_store: SnapshotStore,
    snapshot_frequency: int,
    semaphore: threading.BoundedSemaphore,
    stop_event: threading.Event,
):
    """_populate_queue calls `iter(source)` to get an iterator `it`, waits for semaphore.acquire,
    and puts its outputs onto q. It never releases the sempahore. It continues to put items on the
    q as long as it can acquire the sempahore, stop_event is not set, and StopIteration has not
    been thrown by the `it`.

    This function will always put tuples of (x, idx) on the q where idx
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

    def _put(item, block: bool = True, snapshot: Optional[Dict[str, Any]] = None):
        _idx = idx.get()
        if snapshot:
            snapshot_store.append(snapshot=snapshot, version=_idx)
        q.put((item, _idx), block=block)

    try:
        assert (
            isinstance(snapshot_frequency, int) and snapshot_frequency >= 0
        ), f"snapshot_frequency {snapshot_frequency} must be non-negative integer!"
        src_iter = iter(source)
    except Exception:
        e = StartupExceptionWrapper(where="in _populate_queue startup for device")
        _put(e)
        return

    yielded = 0
    while not stop_event.is_set():
        snapshot = None
        if not semaphore.acquire(blocking=True, timeout=QUEUE_TIMEOUT):
            continue
        try:
            item = next(src_iter)  # FIXME: This may hang!
            yielded += 1
            if snapshot_frequency > 0 and yielded % snapshot_frequency == 0:
                snapshot = source.state_dict()
        except StopIteration as e:
            _put(e)
            break
        except Exception:
            item = ExceptionWrapper(where="in _populate_queue")
        try:
            _put(item, block=False, snapshot=snapshot)  # Semaphore should prevent this from throwing
        except queue.Full:
            raise RuntimeError("Queue should not be full")
