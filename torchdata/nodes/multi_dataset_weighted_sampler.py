# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, Iterator, Optional

import torch
from torchdata.nodes.base_node import BaseNode, T


class MultiDatasetWeightedSampler(BaseNode[T]):
    """A node that samples from multiple datasets with weights."""

    _ARBITRARY_LARGE_NUMBER = 1000

    def __init__(
        self,
        source_nodes: Dict[str, BaseNode[T]],
        weights: Dict[str, float],
        stopping_criterion: str = "CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED",
    ) -> None:
        self.source_nodes = source_nodes
        self.weights = weights
        self.stopping_criterion = stopping_criterion

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        self._it = self.Iter(self, initial_state)
        return self._it

    def get_state(self) -> Dict[str, Any]:
        if self._it is None:
            iter(self)
        return self._it.get_state()

    class Iter(Iterator[T]):
        DATASET_NODE_STATES_KEY = "dataset_node_states"
        GENERATOR_STATE_KEY = "g_state"
        DATASETS_EXHAUSTED_KEY = "_datasets_exhausted"

        def __init__(self, parent, initial_state: Optional[Dict[str, Any]]):
            self._num_rand_list_skip = 0
            self._datasets_exhausted = [False] * len(parent.weights)

            g = torch.Generator()
            g.manual_seed(0)
            self._weighted_sampler = _WeightedSampler(
                weights=parent.weights,
                generator=g,
            )
            if initial_state is not None:
                self._weighted_sampler = _WeightedSampler(
                    weights=parent.weights,
                    generator=g,
                    initial_state=initial_state[self.GENERATOR_STATE_KEY],
                )
                self._datasets_exhausted = initial_state[self.DATASETS_EXHAUSTED_KEY]
                for k in parent.source_nodes.keys():
                    parent.source_nodes[k].load_state_dict(initial_state[self.DATASET_NODE_STATES_KEY][k])
            else:
                # Force a fresh iterator from all source nodes
                for k in parent.source_nodes.keys():
                    parent.source_nodes[k].load_state_dict(None)

            self._source_nodes = parent.source_nodes
            self._ds_iters = [iter(self._source_nodes[k]) for k in self._weighted_sampler.names]

        def __iter__(self) -> Iterator[T]:
            return self

        def __next__(self) -> T:
            if all(self._datasets_exhausted):
                raise StopIteration()

            idx, key = next(self._weighted_sampler)
            try:
                item = next(self._ds_iters[idx])
            except StopIteration:
                self._datasets_exhausted[idx] = True
                if all(self._datasets_exhausted):
                    raise StopIteration()
                self._ds_iters[idx] = iter(self._source_nodes[key])
                item = next(self._ds_iters[idx])

            return item

        def get_state(self) -> Dict[str, Any]:
            return {
                self.GENERATOR_STATE_KEY: self._weighted_sampler.get_state(),
                self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
                self.DATASET_NODE_STATES_KEY: {
                    k: self._source_nodes[k].state_dict() for k in self._weighted_sampler.names
                },
            }


class _WeightedSampler:
    def __init__(
        self,
        weights: Dict[str, float],
        generator,
        batch_size: int = 1000,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        self.names, self.weights = [], []
        for name, weight in weights.items():
            self.names.append(name)
            self.weights.append(weight)
        self.weights = torch.tensor(self.weights, dtype=torch.float64)

        self.batch_size = batch_size
        self._g = generator
        self._g_snapshot = self._g.get_state()
        if initial_state is not None:
            self._g.set_state(initial_state["g_state"])
            self._offset = initial_state["offset"]
        else:
            self._offset = 0

        self._batch_of_indices = self._get_batch_of_indices()

    def _get_batch_of_indices(self) -> list[int]:
        self._g_snapshot = self._g.get_state()
        return torch.multinomial(
            self.weights,
            num_samples=self.batch_size,
            replacement=True,
            generator=self._g,
        ).tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if self._offset >= len(self._batch_of_indices):
            self._batch_of_indices = self._get_batch_of_indices()
            self._offset = 0
        item = self._batch_of_indices[self._offset]
        self._offset += 1
        return item, self.names[item]

    def get_state(self):
        return {
            "g_state": self._g_snapshot,
            "offset": self._offset,
        }
