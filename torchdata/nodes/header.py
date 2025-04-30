# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, TypeVar

from torchdata.nodes import BaseNode


T = TypeVar("T")


class Header(BaseNode[T]):
    """Node that yields only the first N items from source node.

    This node limits the number of items yielded from the source node to at most N.
    After N items have been yielded, it will raise StopIteration on subsequent calls
    to next(), even if the source node has more items available.

    Args:
        source_node (BaseNode[T]): The source node to pull items from.
        n (int): The maximum number of items to yield. Must be non-negative.
    """

    SOURCE_KEY = "source"
    NUM_YIELDED_KEY = "num_yielded"

    def __init__(self, source_node: BaseNode[T], n: int):
        super().__init__()
        if n < 0:
            raise ValueError("n must be non-negative")
        self.source = source_node
        self.n = n
        self._num_yielded = 0  # Count of items yielded so far

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset the node to its initial state or to the provided state.

        Args:
            initial_state: Optional state dictionary to restore from.
        """
        super().reset(initial_state)
        if initial_state is not None:
            # Be strict about required keys in the state
            self.source.reset(initial_state[self.SOURCE_KEY])
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
        else:
            self.source.reset(None)
            self._num_yielded = 0

    def next(self) -> T:
        """Get the next item from the source node if fewer than N items have been yielded.

        Returns:
            The next item from the source node.

        Raises:
            StopIteration: If N items have already been yielded or the source is exhausted.
        """
        if self._num_yielded >= self.n:
            raise StopIteration

        item = next(self.source)
        self._num_yielded += 1
        return item

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the node.

        Returns:
            Dict[str, Any] - A dictionary containing the state of the source node and number of cycles completed.
        """
        return {
            self.SOURCE_KEY: self.source.state_dict(),
            self.NUM_YIELDED_KEY: self._num_yielded,
        }
