# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Dict, Iterator, Literal, Optional, TypeVar

from torchdata.nodes.base_node import BaseNode
from torchdata.nodes.map import Mapper, ParallelMapper

T = TypeVar("T", covariant=True)


class Filter(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[T],
        predicate: Callable[[T], bool],
        num_workers: int = 0,
        in_order: bool = True,
        method: Literal["thread", "process"] = "thread",
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

        else:
            self._it = _InlineFilterIter(
                source=self.source,
                predicate=self.predicate,
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
    """
    An iterator that filters data samples in parallel.

    Args:
        source (BaseNode[T]): The source node providing data samples.
        predicate (Callable[[T], bool]): A function that takes a data sample and returns a boolean indicating whether to include it.
        num_workers (int): The number of worker processes to use for parallel filtering.
        in_order (bool): Whether to preserve the order of data samples.
        method (Literal["thread", "process"]): The method to use for parallelization.
        multiprocessing_context (Optional[str]): The multiprocessing context to use.
        max_concurrent (Optional[int]): The maximum number of concurrent tasks.
        snapshot_frequency (int): The frequency at which to take snapshots.
        initial_state (Optional[Dict[str, Any]]): The initial state to start with.
    """

    def __init__(
        self,
        source: BaseNode[T],
        predicate: Callable[[T], bool],
        num_workers: int,
        in_order: bool,
        method: Literal["thread", "process"],
        multiprocessing_context: Optional[str],
        max_concurrent: Optional[int],
        snapshot_frequency: int,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.predicate = predicate
        self.num_workers = num_workers
        self.in_order = in_order
        self.method = method
        self.multiprocessing_context = multiprocessing_context
        self.max_concurrent = max_concurrent
        self.snapshot_frequency = snapshot_frequency
        # Create a ParallelMapper to filter items in parallel
        self.mapper = ParallelMapper(
            source=self.source,
            map_fn=lambda x: (x, self.predicate(x)),
            num_workers=self.num_workers,
            in_order=self.in_order,
            method=self.method,
            multiprocessing_context=self.multiprocessing_context,
            max_concurrent=self.max_concurrent,
            snapshot_frequency=self.snapshot_frequency,
        )
        if initial_state is not None:
            self.mapper.reset(initial_state)

    def __iter__(self) -> Iterator[T]:
        """
        Returns the iterator object itself.

        Returns:
            Iterator[T]: The iterator object itself.
        """
        return self

    def __next__(self) -> T:
        """
        Returns the next filtered data sample.

        Returns:
            T: The next filtered data sample.
        """
        while True:
            try:
                item, passed_predicate = next(self.mapper)
                if passed_predicate:
                    return item
            except StopIteration:
                raise

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the parallel filter iterator.

        Returns:
            Dict[str, Any]: The current state of the parallel filter iterator.
        """
        return self.mapper.get_state()

    def __del__(self):
        # Clean up resources when the iterator is deleted
        del self.mapper
