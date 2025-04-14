# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Any, Dict, Mapping, Optional

from torchdata.nodes.base_node import BaseNode, T
from torchdata.nodes.samplers.stop_criteria import StopCriteria


class MultiNodeRoundRobinSampler(BaseNode[T]):
    """A node that samples from multiple datasets in a round robin fashion.
    This node expects to take in a dictionary of source nodes.
    The node implements the state using the following keys:
    - CURRENT_DATASET_INDEX_KEY: The index of the current dataset.
    - DATASET_NODE_STATES_KEY: A dictionary of states for each source node.
    - DATASETS_EXHAUSTED_KEY: A dictionary of booleans indicating whether each source node is exhausted.

    We support multiple stopping criteria:
    - CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED: Cycle through the source nodes until all datasets
        are exhausted. This is the default behavior.
    - FIRST_DATASET_EXHAUSTED: Stop when the first dataset is exhausted.
    - ALL_DATASETS_EXHAUSTED: Stop when all datasets are exhausted.
    Exhaustion Handling:
        A dataset is only marked exhausted when it raises StopIteration.
        Example: With 4 nodes (1st has 1 item, others have 5):
        - 1st iteration: Yields 1st node's item (not yet exhausted)
        - Subsequent iterations: Yields from other nodes
        - When returning to 1st node, StopIteration marks it exhausted
        - Behavior depends on criteria: FIRST_DATASET_EXHAUSTED stops immediately, others continue

    Args:
        source_nodes (Mapping[str, BaseNode[T]]): A dictionary of source nodes.
        stop_criteria (str): The stopping criteria. Default is CYCLE_UNTIL_ALL_DATASETS_EXHAUST.

    Example:
        >>> # Dataset A: 1 element, Dataset B: 2 elements
        >>> sampler = MultiNodeRoundRobinSampler(
        ...     source_nodes={"A": A_node, "B": B_node},
        ...     stop_criteria=StopCriteria.FIRST_DATASET_EXHAUSTED
        ... )
        >>> list(sampler)  # Yields: A, B, then A is exhausted
        [A_item, B_item1]
        If using StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED:
        >>> list(sampler)  # Yields: A, B,  A (exhausted), B , A, then B is exhausted
        [A_item, B_item1, A_item, B_item2, A_item ]
    """

    CURRENT_DATASET_INDEX_KEY = "current_dataset_index"
    DATASET_NODE_STATES_KEY = "dataset_node_states"
    DATASETS_EXHAUSTED_KEY = "datasets_exhausted"

    def __init__(
        self,
        source_nodes: Mapping[str, BaseNode[T]],
        stop_criteria: str = StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
    ) -> None:
        super().__init__()
        self.source_nodes = [source_nodes[k] for k in source_nodes.keys()]
        self.num_datasets = len(self.source_nodes)
        self.stop_criteria = stop_criteria
        self._current_dataset_index = 0
        self._validate_stop_criteria()
        self._datasets_exhausted = [False for _ in range(self.num_datasets)]

    def _validate_stop_criteria(self) -> None:
        if self.stop_criteria not in [
            StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            StopCriteria.ALL_DATASETS_EXHAUSTED,
            StopCriteria.FIRST_DATASET_EXHAUSTED,
        ]:
            raise ValueError(
                f"Invalid {self.stop_criteria=}. stop_criteria must be one of: CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, ALL_DATASETS_EXHAUSTED, FIRST_DATASET_EXHAUSTED"
            )

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self._datasets_exhausted = initial_state[self.DATASETS_EXHAUSTED_KEY]
            for k in range(self.num_datasets):
                self.source_nodes[k].reset(initial_state[self.DATASET_NODE_STATES_KEY][k])
            self._current_dataset_index = initial_state[self.CURRENT_DATASET_INDEX_KEY]
        else:
            # Force a fresh iterator from all source nodes
            self._datasets_exhausted = [False for _ in range(self.num_datasets)]
            self._current_dataset_index = 0
            for k in range(self.num_datasets):
                self.source_nodes[k].reset()

    def _check_for_stop_iteration(self) -> None:
        if all(self._datasets_exhausted):
            # Raise StopIteration if all datasets are exhausted,
            # this covers for both ALL_DATASETS_EXHAUSTED and CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
            raise StopIteration()
        # Raise StopIteration is StopCriteria is FIRST_DATASET_EXHAUSTED and
        # the first dataset is exhausted. Doing this to correctly catch StopIteration
        # when trying next(it) on already exhausted iterator
        if self.stop_criteria == StopCriteria.FIRST_DATASET_EXHAUSTED and any(self._datasets_exhausted):
            raise StopIteration()
        return

    def next(self) -> T:
        while True:
            self._check_for_stop_iteration()
            current_iterator = self.source_nodes[self._current_dataset_index]
            try:
                # Before fetching a new item check if the current dataset is already
                # exhaused and if StopCriteria is ALL_DATASETS_EXHAUSTED, move to next dataset
                if (
                    self._datasets_exhausted[self._current_dataset_index]
                    and self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED
                ):
                    self._current_dataset_index = (self._current_dataset_index + 1) % self.num_datasets
                    continue
                item = next(current_iterator)
            except StopIteration:
                # Mark the dataset as exhausted
                self._datasets_exhausted[self._current_dataset_index] = True
                # Based on updated _check_for_stop_iteration, check if we should raise StopIteration
                self._check_for_stop_iteration()
                # If StopCriteria is ALL_DATASETS_EXHAUSTED, move to next dataset
                if self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED:
                    continue
                # If StopCriteria is CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
                # reset the iterator and try again
                self.source_nodes[self._current_dataset_index].reset()
                item = next(self.source_nodes[self._current_dataset_index])
            break
        # If we did't throw StopIteration, increment the number of items yielded and return the item
        self._current_dataset_index = (self._current_dataset_index + 1) % self.num_datasets
        return item

    def get_state(self) -> Dict[str, Any]:
        state = {
            self.CURRENT_DATASET_INDEX_KEY: self._current_dataset_index,
            self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
            self.DATASET_NODE_STATES_KEY: {k: self.source_nodes[k].state_dict() for k in range(self.num_datasets)},
        }
        return state
