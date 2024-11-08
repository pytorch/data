# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, Mapping, Optional

import torch
from torchdata.nodes.base_node import BaseNode, T


class MultiDatasetWeightedSampler(BaseNode[T]):
    """A node that samples from multiple datasets with weights."""

    DATASET_NODE_STATES_KEY = "dataset_node_states"

    _ARBITRARY_LARGE_NUMBER = 100

    def __init__(
        self,
        data_nodes: Mapping[str, BaseNode[T]],
        weights: Dict[str, float],
        stopping_criterion: str = "CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED",
    ) -> None:
        self.data_nodes = data_nodes
        self.weights = weights
        self.stopping_criterion = stopping_criterion

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        """Logic for iterating over multiple datasets with weights"""
        if initial_state is not None:  # TODO add more checks here
            # load state from initial_state
            dataset_node_state = initial_state[self.DATASET_NODE_STATES_KEY]
            # load state for each parent dataset node
            for dataset_key in self.data_nodes.keys():
                self.data_nodes[dataset_key].load_state_dict(dataset_node_state[dataset_key])

        # create iterators for each dataset node
        dataset_iterators = {
            dataset_idx: iter(dataset_node) for dataset_idx, dataset_node in enumerate(self.data_nodes.values())
        }
        dataset_exhausted = [False] * len(self.weights)

        # create a weighted sampler
        g = torch.Generator()
        g.manual_seed(0)  # TODO: incorporate seed from worker_info, epoch
        while all(dataset_exhausted) is False:
            weighted_sampler = torch.multinomial(
                torch.tensor(list(self.weights.values())),
                num_samples=self._ARBITRARY_LARGE_NUMBER,
                replacement=True,
                generator=g,
            ).tolist()

            # iterate over the weighted sampler
            for dataset_idx in weighted_sampler:
                # yield from the iterator for the dataset
                try:
                    if dataset_exhausted[dataset_idx]:
                        continue  # skip this dataset and move on to the next one
                    yield next(dataset_iterators[dataset_idx])
                except (StopIteration):  # TODO: handle the case where all datasets are exhausted | StopIteration
                    # mark the dataset as exhausted
                    dataset_exhausted[dataset_idx] = True

    def get_state(self) -> Dict[str, Any]:
        # TODO: add more keys here
        return {
            self.DATASET_NODE_STATES_KEY: {
                dataset_key: dataset_node.state_dict() for dataset_key, dataset_node in self.data_nodes.items()
            }
        }
