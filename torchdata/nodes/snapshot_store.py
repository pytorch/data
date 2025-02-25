import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

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

    def get_initial_snapshot(self, thread: threading.Thread, timeout: float) -> Any:
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
            while self._q.queue and version >= self._q.queue[0][0]:
                ver, val = self._q.get_nowait()

        if ver == version:
            return val
        else:
            return None

    def append_initial_snapshot(self, snapshot: Any) -> None:
        self.append(snapshot, self.SNAPSHOT_INIT_VERSION)

    def get_initial_snapshot(self, thread: threading.Thread, timeout: float = 60.0) -> Any:
        snapshot = None
        ver = None

        ack_t0 = time.time()
        while snapshot is None and time.time() - ack_t0 < timeout:
            try:
                ver, snapshot = self._q.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                pass
            if not thread.is_alive():
                # Don't test this until after QUEUE_TIMEOUT has elapsed because
                # thread may inadvertently report "is_alive()==False"
                break

        if snapshot is not None and isinstance(snapshot, ExceptionWrapper):
            snapshot.reraise()

        if snapshot is None or ver != self.SNAPSHOT_INIT_VERSION:
            raise RuntimeError(
                f"Failed to get initial snapshot after {time.time() - ack_t0} seconds! {thread.is_alive()=}, {snapshot=}, {ver=}"
            )

        return snapshot
