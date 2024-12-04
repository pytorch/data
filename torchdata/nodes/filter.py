# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Dict, Iterator, Optional

from torchdata.nodes.base_node import BaseNode, T


class Filter(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[T],
        predicate: Callable[[T], bool],
        num_workers: int = 0,
        in_order: bool = True,
        method: str = "thread",
        multiprocessing_context: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        snapshot_frequency: int = 1,
    ):
        super().__init__()
        self.source = source
        self.predicate = predicate
        self.num_workers = num_workers
        self.in_order = in_order
        self.method = method
        self.multiprocessing_context = multiprocessing_context
        self.max_concurrent = max_concurrent
        self.snapshot_frequency = snapshot_frequency
        self._it: Optional[Iterator[T]] = None

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if self._it is not None:
            del self._it
        if self.num_workers > 0:
            self._parallel_reset(initial_state)
        else:
            self._inline_reset(initial_state)

    def _inline_reset(self, initial_state: Optional[Dict[str, Any]]):
        self._it = _InlineFilterIter(
            source=self.source,
            predicate=self.predicate,
            initial_state=initial_state,
        )

    def _parallel_reset(self, initial_state: Optional[Dict[str, Any]]):
        self._it = _ParallelFilterIter(
            source=self.source,
            predicate=self.predicate,
            num_workers=self.num_workers,
            in_order=self.in_order,
            method=self.method,
            multiprocessing_context=self.multiprocessing_context,
            max_concurrent=self.max_concurrent,
            snapshot_frequency=self.snapshot_frequency,
            initial_state=initial_state,
        )

    def next(self):
        return next(self._it)  # type: ignore[arg-type]

    def get_state(self) -> Dict[str, Any]:
        return self._it.get_state()  # type: ignore[union-attr]


class _InlineFilterIter(Iterator[T]):
    def __init__(
        self,
        source: BaseNode[T],
        predicate: Callable[[T], bool],
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.predicate = predicate
        if initial_state is not None:
            self.source.reset(initial_state["source"])
        else:
            self.source.reset()

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        while True:
            item = next(self.source)
            if self.predicate(item):
                return item

    def get_state(self) -> Dict[str, Any]:
        return {"source": self.source.state_dict()}


class _ParallelFilterIter(Iterator[T]):
    def __init__(
        self,
        source: BaseNode[T],
        predicate: Callable[[T], bool],
        num_workers: int,
        in_order: bool,
        method: str,
        multiprocessing_context: Optional[str],
        max_concurrent: Optional[int],
        snapshot_frequency: int,
        initial_state: Optional[Dict[str, Any]],
    ):
        self.source = source
        self.predicate = predicate
        self.num_workers = num_workers
        self.in_order = in_order
        self.method = method
        self.multiprocessing_context = multiprocessing_context
        self.max_concurrent = max_concurrent
        self.snapshot_frequency = snapshot_frequency
        self._in_q: queue.Queue = queue.Queue()
        self._out_q: queue.Queue = queue.Queue()
        self._sem = threading.BoundedSemaphore(value=max_concurrent or 2 * num_workers)
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        for _ in range(num_workers):
            t = threading.Thread(
                target=self._filter_worker,
                args=(self._in_q, self._out_q, self.predicate),
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        self._populate_queue_thread = threading.Thread(
            target=_populate_queue,
            args=(
                self.source,
                self._in_q,
                QueueSnapshotStore(),
                snapshot_frequency,
                self._sem,
                self._stop_event,
            ),
            daemon=True,
        )

        self._populate_queue_thread.start()
        if initial_state is not None:
            self.source.reset(initial_state["source"])
        else:
            self.source.reset()

    def _filter_worker(
        self, in_q: queue.Queue, out_q: queue.Queue, predicate: Callable[[T], bool]
    ) -> None:
        while True:
            try:
                item = in_q.get(block=True, timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            if isinstance(item, StopIteration):
                out_q.put(item)
                break
            elif predicate(item):
                out_q.put(item)
            self._sem.release()

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        while True:
            try:
                item = self._out_q.get(block=True, timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    raise StopIteration()
                continue
            if isinstance(item, StopIteration):
                raise item
            return item

    def get_state(self) -> Dict[str, Any]:
        return {"source": self.source.state_dict()}

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        self._stop_event.set()
        if (
            hasattr(self, "_populate_queue_thread")
            and self._populate_queue_thread.is_alive()
        ):
            self._populate_queue_thread.join(timeout=0.5)
        for t in self._workers:
            if t.is_alive():
                t.join(timeout=0.5)
