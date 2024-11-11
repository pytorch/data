# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, Iterator, Mapping, Optional

import torch
from torchdata.nodes.base_node import BaseNode, T


class MultiDatasetWeightedSampler(BaseNode[T]):
    """A node that samples from multiple datasets with weights."""

    DATASET_NODE_STATES_KEY = "dataset_node_states"
    NUM_YIELDED_KEY = "_num_yielded"
    GENERATOR_STATE_KEY = "g_state"
    NUM_RAND_LIST_SKIP_KEY = "_num_rand_list_skip"
    DATASETS_EXHAUSTED_KEY = "_datasets_exhausted"

    _ARBITRARY_LARGE_NUMBER = 1000

    def __init__(
        self,
        data_nodes: Mapping[str, BaseNode[T]],
        weights: Dict[str, float],
        stopping_criterion: str = "CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED",
    ) -> None:
        self.data_nodes = data_nodes
        self.weights = weights
        self.stopping_criterion = stopping_criterion
        self._num_yielded = 0
        self._num_rand_list_skip = 0
        self._datasets_exhausted = [False] * len(self.weights)

        # create a weighted sampler
        self.g = torch.Generator()
        self.g.manual_seed(0)  # TODO: incorporate seed from worker_info, epoch

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        """Logic for iterating over multiple datasets with weights"""

        if initial_state is not None:  # TODO add more checks here
            # load state from initial_state
            dataset_node_state = initial_state[self.DATASET_NODE_STATES_KEY]
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
            self._num_rand_list_skip = initial_state[self.NUM_RAND_LIST_SKIP_KEY]
            self._datasets_exhausted = initial_state[self.DATASETS_EXHAUSTED_KEY]
            self.g.set_state(initial_state[self.GENERATOR_STATE_KEY])
            # batch_of_indices = self._generate_indices(self.g)

            # self._idx_into_batch_of_indices = initial_state["idx_into_batch_of_indices"]

            # load state for each parent dataset node
            for dataset_key in self.data_nodes.keys():
                self.data_nodes[dataset_key].load_state_dict(
                    dataset_node_state[dataset_key],
                )

        # create iterators for each dataset node
        dataset_iterators = {
            dataset_idx: iter(self.data_nodes[dataset_key])
            for dataset_idx, dataset_key in enumerate(self.data_nodes.keys())
        }

        while not all(self._datasets_exhausted):

            # if self._idx_into_batch_of_indices >= len(batch_of_indices):
            #     # self._g_state_snapshot = self.g.get_state()
            #     batch_of_indices = self._generate_indices(self.g)
            #     self._idx_into_batch_of_indices = 0

            # dataset_idx = batch_of_indices[self._idx_into_batch_of_indices]
            # self._idx_into_batch_of_indices += 1

            # ...

            # yield item

            weighted_sampler = torch.multinomial(
                torch.tensor(list(self.weights.values())),
                num_samples=self._ARBITRARY_LARGE_NUMBER,
                replacement=True,
                generator=self.g,
            ).tolist()

            # iterate over the weighted sampler
            fast_forward_counter = self._num_yielded
            for dataset_idx in weighted_sampler[fast_forward_counter:]:
                # yield from the iterator for the dataset
                try:
                    if all(self._datasets_exhausted):
                        break
                    self._num_yielded += 1
                    yield next(dataset_iterators[dataset_idx])
                except (StopIteration):  # TODO: handle the case where all datasets are exhausted | StopIteration
                    self._datasets_exhausted[dataset_idx] = True

                    dataset_iterators[dataset_idx] = iter(
                        self.data_nodes[list(self.data_nodes.keys())[dataset_idx]]
                    )  # reset the iterator for the dataset
                    yield next(dataset_iterators[dataset_idx])
            fast_forward_counter = 0

    def get_state(self) -> Dict[str, Any]:
        # TODO: add more keys here
        return {
            self.NUM_YIELDED_KEY: self._num_yielded,
            self.GENERATOR_STATE_KEY: self.g.get_state(),
            self.NUM_RAND_LIST_SKIP_KEY: self._num_rand_list_skip,
            self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
            self.DATASET_NODE_STATES_KEY: {
                dataset_key: self.data_nodes[dataset_key].state_dict() for dataset_key in self.data_nodes.keys()
            },
        }
