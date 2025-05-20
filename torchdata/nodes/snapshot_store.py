# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union

from torchdata.nodes.constants import QUEUE_TIMEOUT

from torchdata.nodes.exception_wrapper import ExceptionWrapper


@dataclass
class MonotonicIndex:
    initial: int = 0

    def __post_init__(self):
        self._idx = self.initial

    def get(self) -> int:
        idx = self._idx
        self._idx += 1
        return idx


class SnapshotStore(Protocol):
    """Protocol for passing snapshot state around between threads and processes"""

    def append(self, snapshot: Any, version: int):
        ...

    def pop_version(self, version: int) -> Optional[Any]:
        ...

    def append_initial_snapshot(self, snapshot: Any):
        ...

    def get_initial_snapshot(self, thread: Union[Future, threading.Thread], timeout: float) -> Any:
        ...


class QueueSnapshotStore(SnapshotStore):
    """A snapshot store that uses a queue to store snapshots"""

    SNAPSHOT_INIT_VERSION = -1

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._max_version: int = -1000

    def append(self, snapshot: Any, version: int) -> None:
        with self._lock:
            if version <= self._max_version:
                raise ValueError(f"{version=} is not strictly greater than {self._max_version=}")
            self._max_version = version
            self._q.put((version, snapshot))

    def pop_version(self, version: int) -> Optional[Any]:
        ver, val = None, None
        with self._lock:
            # pop all items that have a lesser version index
            while self._q.queue and version >= self._q.queue[0][0]:
                ver, val = self._q.get_nowait()

        if ver == version:
            return val
        else:
            return None

    def append_initial_snapshot(self, snapshot: Any) -> None:
        self.append(snapshot, self.SNAPSHOT_INIT_VERSION)

    def get_initial_snapshot(self, thread: Union[Future, threading.Thread], timeout: float = 60.0) -> Any:
        snapshot = None
        ver = None

        ack_t0 = time.time()
        while snapshot is None and time.time() - ack_t0 < timeout:
            try:
                ver, snapshot = self._q.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                pass
            # Don't test this until after QUEUE_TIMEOUT has elapsed because
            # thread may inadvertently report "is_alive()==False"
            if isinstance(thread, Future) and not thread.running():
                break
            if isinstance(thread, threading.Thread) and not thread.is_alive():
                break

        if snapshot is not None and isinstance(snapshot, ExceptionWrapper):
            snapshot.reraise()

        if snapshot is None or ver != self.SNAPSHOT_INIT_VERSION:
            error_msg = thread.is_alive() if isinstance(thread, threading.Thread) else thread.running()
            raise RuntimeError(
                f"Failed to get initial snapshot after {time.time() - ack_t0} seconds! thread.is_alive()={error_msg} {snapshot=}, {ver=}"
            )

        return snapshot
