# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, Optional

import torch
from torchdata.nodes.base_node import BaseNode, T


class MultiDatasetWeightedSampler(BaseNode[T]):
    """A node that samples from multiple datasets with weights."""

    DATASET_NODE_STATES_KEY = "dataset_node_states"

    def __init__(self, data_nodes: Dict[str, BaseNode[T]], weights: Dict[str, float]) -> None:
        self.data_nodes = data_nodes
        self.weights = weights

    def iterator(self, initial_state: Optional[dict]) -> Iterator[T]:
        """Logic for iterating over multiple datasets with weights"""
        if initial_state is not None:  # TODO add more checks here
            # load state from initial_state
            dataset_node_states = initial_state[self.DATASET_NODE_STATES_KEY]
            # load state for each parent dataset node
            for dataset_key, dataset_node in self.data_nodes.items():
                dataset_node.load_state_dict(dataset_node_states[dataset_key])

        # create iterators for each dataset node
        dataset_iterators = {
            dataset_idx: iter(dataset_node) for dataset_idx, dataset_node in enumerate(self.data_nodes.values())
        }

        # create a weighted sampler
        g = torch.Generator()
        g.manual_seed(0)  # TODO: incorporate seed from worker_info, epoch, iteration
        while True:
            weighted_sampler = torch.multinomial(
                torch.tensor(list(self.weights.values())),
                num_samples=len(self.weights),
                replacement=True,
                generator=g,
            )

            # iterate over the weighted sampler
            for dataset_idx in weighted_sampler:
                # return the iterator for the dataset
                return dataset_iterators[dataset_idx]

            # TODO: handle the case where all datasets are exhausted | StopIteration

    def get_state(self) -> Dict[str, Any]:
        # TODO: add more keys here
        return {
            self.DATASET_NODE_STATES_KEY: {
                dataset_key: dataset_node.state_dict() for dataset_key, dataset_node in self.data_nodes.items()
            }
        }
