# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, TypeVar

from torchdata.nodes import BaseNode


T = TypeVar("T")


class Cycler(BaseNode[T]):
    """Node that cycles through source node a limited or unlimited number of times.

    This node will continuously loop through the source node. When the source node
    is exhausted, it will be reset and iteration will start from the beginning again.
    The node keeps track of how many times it has completed a full cycle through
    the source and the total number of items yielded.

    Args:
        source_node (BaseNode[T]): The source node to cycle through.
        max_cycles (Optional[int]): Maximum number of cycles to perform. If None,
            cycles indefinitely. Must be positive if specified. Default: None.
    """

    SOURCE_KEY = "source"
    NUM_CYCLES_KEY = "num_cycles"
    HAS_STARTED_KEY = "has_started"
    NUM_YIELDED_KEY = "num_yielded"
    MAX_CYCLES_KEY = "max_cycles"

    def __init__(self, source_node: BaseNode[T], max_cycles: Optional[int] = None):
        super().__init__()
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be positive if specified")

        self.source = source_node
        self.max_cycles = max_cycles
        self._num_cycles = 0
        self._has_started = False
        self._num_yielded = 0

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset the node to its initial state or to the provided state.

        Args:
            initial_state: Optional state dictionary to restore from.
        """
        super().reset(initial_state)
        if initial_state is not None:
            # Be strict about required keys in the state
            self._num_cycles = initial_state[self.NUM_CYCLES_KEY]
            self._has_started = initial_state[self.HAS_STARTED_KEY]
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
            self.max_cycles = initial_state[self.MAX_CYCLES_KEY]
            self.source.reset(initial_state[self.SOURCE_KEY])
        else:
            self._num_cycles = 0
            self._has_started = False
            self._num_yielded = 0
            self.source.reset(None)

    def next(self) -> T:
        """Get the next item from the source node, cycling if necessary.

        Returns:
            The next item from the source node.

        Raises:
            StopIteration: If the source node is empty or max_cycles is reached.
        """
        try:
            item = next(self.source)
            self._has_started = True
            self._num_yielded += 1
            return item
        except StopIteration:
            # If this is the first time we're trying to get an item and it fails,
            # the source is empty - just propagate the StopIteration without cycling
            if not self._has_started:
                raise StopIteration

            # Otherwise, source is exhausted after yielding some items
            # Increment cycle count and check max_cycles limit
            self._num_cycles += 1

            # If we've reached max_cycles, stop iteration
            if self.max_cycles is not None and self._num_cycles >= self.max_cycles:
                raise StopIteration

            # Reset source and continue
            self.source.reset(None)

            # Try to get the first item after reset
            # This could still raise StopIteration if the source becomes empty
            # after reset (e.g., a dynamic source that changes over time)
            try:
                item = next(self.source)
                self._num_yielded += 1
                return item
            except StopIteration:
                raise

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the node.

        Returns:
            Dict[str, Any]: A dictionary containing the state of the source node,
            number of cycles completed, whether iteration has started,
            total number of items yielded, and the maximum number of cycles.
        """
        return {
            self.SOURCE_KEY: self.source.state_dict(),
            self.NUM_CYCLES_KEY: self._num_cycles,
            self.HAS_STARTED_KEY: self._has_started,
            self.NUM_YIELDED_KEY: self._num_yielded,
            self.MAX_CYCLES_KEY: self.max_cycles,
        }
