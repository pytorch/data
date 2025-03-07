import random
from collections import deque
from typing import Any, Deque, Dict, Optional, TypeVar

from torchdata.nodes import BaseNode


T = TypeVar("T")


class Shuffler(BaseNode[T]):
    """Node that shuffles items from source node using a buffer.

    This node maintains a buffer of items from the source node and returns them
    in a random order. The buffer is filled to capacity initially and then
    replenished as items are removed. The randomization is controlled by a
    random number generator with an optional seed for reproducibility.

    Args:
        source_node (BaseNode[T]): The source node to pull items from.
        buffer_size (int): Size of the buffer used for shuffling. Must be at least 1.
        seed (Optional[int]): Optional seed for the random number generator.
    """

    SOURCE_KEY = "source"
    RNG_STATE_KEY = "rng_state"
    BUFFER_KEY = "buffer"
    NUM_SHUFFLED_KEY = "num_shuffled"
    RANDOM_STATE_KEY = "random_state"

    def __init__(self, source_node: BaseNode[T], buffer_size: int, seed: Optional[int] = None):
        super().__init__()
        if buffer_size < 1:
            raise ValueError("Buffer size must be at least 1")
        self.source = source_node
        self.buffer_size = buffer_size
        self.buffer: Deque[T] = deque()
        self.rng = random.Random(seed)
        self._initial_seed = seed
        self._num_shuffled = 0

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset the node to its initial state or to the provided state.

        Args:
            initial_state: Optional state dictionary to restore from.
        """
        super().reset(initial_state)
        self.buffer.clear()
        self._num_shuffled = 0

        if initial_state is not None:
            self.source.reset(initial_state.get(self.SOURCE_KEY))
            self.rng.setstate(initial_state.get(self.RNG_STATE_KEY))
            self.buffer = initial_state.get(self.BUFFER_KEY, [])
            self._num_shuffled = initial_state.get(self.NUM_SHUFFLED_KEY, 0)
        else:
            self.source.reset(None)
            if self._initial_seed is not None:
                self.rng = random.Random(self._initial_seed)

    def _fill_buffer(self) -> bool:
        """Fill buffer with items from source.

        Returns:
            True if any items were added to the buffer, False otherwise.
        """
        try:
            while len(self.buffer) < self.buffer_size:
                self.buffer.append(next(self.source))
            return True
        except StopIteration:
            return len(self.buffer) > 0

    def next(self) -> T:
        """Get the next item from the buffer in random order.

        Returns:
            A randomly selected item from the buffer.

        Raises:
            StopIteration: If there are no more items in the buffer and the source is exhausted.
        """
        if not self.buffer and not self._fill_buffer():
            raise StopIteration

        # Randomly select and remove an item from the buffer
        idx = self.rng.randrange(len(self.buffer))
        item = self.buffer[idx]
        self.buffer[idx] = self.buffer[-1]
        self.buffer.pop()

        # Try to refill buffer
        self._fill_buffer()
        self._num_shuffled += 1
        return item

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the node.

        Returns:
            A dictionary containing the state of the source node, buffer,
            number of items shuffled, and random state.
        """
        return {
            self.SOURCE_KEY: self.source.state_dict(),
            self.RNG_STATE_KEY: self.rng.getstate(),
            self.BUFFER_KEY: list(self.buffer),
            self.NUM_SHUFFLED_KEY: self._num_shuffled,
            self.RANDOM_STATE_KEY: self.rng.getstate(),
        }
