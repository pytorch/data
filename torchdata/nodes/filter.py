from typing import Any, Callable, Dict, TypeVar, Optional
from torchdata.nodes import BaseNode


T = TypeVar("T")


class Filter(BaseNode[T]):
    """Node that filters items from source node based on predicate function.

    Args:
        source_node (BaseNode[T]): The source node to filter items from.
        filter_fn (Callable[[T], bool]): A function that takes an item and returns True if the item
            should be included, False otherwise.
    """

    SOURCE_KEY = "source"

    def __init__(self, source_node: BaseNode[T], filter_fn: Callable[[T], bool]):
        super().__init__()
        self.source = source_node
        self.filter_fn = filter_fn

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        self.source.reset(initial_state.get(self.SOURCE_KEY) if initial_state else None)

    def next(self) -> T:
        while True:
            item = next(self.source)
            if self.filter_fn(item):
                return item

    def get_state(self) -> Dict[str, Any]:
        return {self.SOURCE_KEY: self.source.state_dict()}
