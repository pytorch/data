from typing import Any, Callable, Dict, Iterator, Literal, Optional, TypeVar
from torchdata.nodes.base_node import BaseNode
from torchdata.nodes.map import ParallelMapper
T = TypeVar("T", covariant=True)

class Filter(BaseNode[T]):
    """
    A node that filters data samples based on a given predicate.
    Args:
        source (BaseNode[T]): The source node providing data samples.
        predicate (Callable[[T], bool]): A function that takes a data sample and returns a boolean indicating whether to include it.
        num_workers (int): The number of worker processes to use for parallel filtering. Defaults to 0.
        in_order (bool): Whether to return items in the order from which they arrive from. Default is True.
        method (Literal["thread", "process"]): The method to use for parallel processing. Default is "thread".
        multiprocessing_context (Optional[str]): The multiprocessing context to use for parallel processing. Default is None.
        max_concurrent (Optional[int]): The maximum number of items to process at once. Default is None.
        snapshot_frequency (int): The frequency at which to snapshot the state of the source node. Default is 1.
    """
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
            )
        else:
            self._it = _InlineFilterIter(source=self.source, predicate=self.predicate)
    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Resets the filter node to its initial state."""
        super().reset(initial_state)
        if self._it is not None:
            self._it.reset(initial_state)
    def next(self) -> T:
        """Returns the next filtered item."""
        return next(self._it)
    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the filter node."""
        return self._it.get_state()

class _InlineFilterIter(Iterator[T]):
    """
    An iterator that filters data samples inline.
    Args:
        source (BaseNode[T]): The source node providing data samples.
        predicate (Callable[[T], bool]): A function that takes a data sample and returns a boolean indicating whether to include it.
    """
    SOURCE_KEY = "source"

    def __init__(self, source: BaseNode[T], predicate: Callable[[T], bool]) -> None:
        self.source = source
        self.predicate = predicate

    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Resets the inline filter iterator to its initial state."""
        if initial_state:
            self.source.reset(initial_state[self.SOURCE_KEY])
        else:
            self.source.reset()

    def __iter__(self) -> Iterator[T]:
        """Returns the iterator object itself."""
        return self
    def __next__(self) -> T:
        """Returns the next filtered item."""
        while True:
            try:
                item = next(self.source)
                if self.predicate(item):
                    return item
            except StopIteration:
                raise
    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the inline filter iterator."""
        return {self.SOURCE_KEY: self.source.state_dict()}

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
    """
    MAPPER_KEY = "mapper"
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
    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Resets the parallel filter iterator to its initial state."""
        if initial_state:
            self.mapper.reset(initial_state[self.MAPPER_KEY])
        else:
            self.mapper.reset()
    def __iter__(self) -> Iterator[T]:
        """Returns the iterator object itself."""
        return self
    def __next__(self) -> T:
        """Returns the next filtered item."""
        while True:

            item, passed_predicate = next(self.mapper)
            if passed_predicate:
                return item

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the parallel filter iterator."""
        return {self.MAPPER_KEY: self.mapper.get_state()}
    def __del__(self):
        # Clean up resources when the iterator is deleted
        del self.mapper
