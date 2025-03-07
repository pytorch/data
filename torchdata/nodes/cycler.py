from typing import Any, Dict, Optional, TypeVar

from torchdata.nodes import BaseNode


T = TypeVar("T")


class Cycler(BaseNode[T]):
    """Node that cycles through source node indefinitely.

    This node will continuously loop through the source node. When the source node
    is exhausted, it will be reset and iteration will start from the beginning again.
    The node keeps track of how many times it has completed a full cycle through
    the source.

    Args:
        source_node (BaseNode[T]): The source node to cycle through.
    """

    SOURCE_KEY = "source"
    NUM_CYCLES_KEY = "num_cycles"
    HAS_STARTED_KEY = "has_started"

    def __init__(self, source_node: BaseNode[T]):
        super().__init__()
        self.source = source_node
        self._num_cycles = 0
        self._has_started = False

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset the node to its initial state or to the provided state.

        Args:
            initial_state: Optional state dictionary to restore from.
        """
        super().reset(initial_state)
        if initial_state is not None:
            self._num_cycles = initial_state.get(self.NUM_CYCLES_KEY, 0)
            self._has_started = initial_state.get(self.HAS_STARTED_KEY, False)
            self.source.reset(initial_state.get(self.SOURCE_KEY))
        else:
            self._num_cycles = 0
            self._has_started = False
            self.source.reset(None)

    def next(self) -> T:
        """Get the next item from the source node, cycling if necessary.

        Returns:
            The next item from the source node.

        Raises:
            StopIteration: If the source node is empty.
        """
        try:
            item = next(self.source)
            self._has_started = True
            return item
        except StopIteration:
            # If this is the first time we're trying to get an item and it fails,
            # the source is empty - just propagate the StopIteration without cycling
            if not self._has_started:
                raise StopIteration

            # Otherwise, source is exhausted after yielding some items
            # Increment cycle count and reset
            self._num_cycles += 1
            self.source.reset(None)

            # Try again - if it's empty, this will raise StopIteration
            return next(self.source)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the node.

        Returns:
            A dictionary containing the state of the source node and number of cycles completed.
        """
        return {
            self.SOURCE_KEY: self.source.state_dict(),
            self.NUM_CYCLES_KEY: self._num_cycles,
            self.HAS_STARTED_KEY: self._has_started,
        }
