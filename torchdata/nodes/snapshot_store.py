import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Protocol


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


class DequeSnapshotStore(SnapshotStore):
    """A snapshot store that uses a deque to store snapshots"""

    def __init__(self, max_size: Optional[int] = None) -> None:
        self._deque: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._max_version: int = -1

    def append(self, snapshot: Any, version: int) -> None:
        with self._lock:
            if version <= self._max_version:
                raise ValueError(f"{version=} is not strictly greater than {self._max_version=}")
            self._max_version = version
            self._deque.append((version, snapshot))

    def pop_version(self, version: int) -> Optional[Any]:
        with self._lock:
            ver, val = None, None
            while self._deque and version >= self._deque[0][0]:
                ver, val = self._deque.popleft()

            if ver == version:
                return val
            else:
                return None
