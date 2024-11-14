# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, Mapping, Optional

import torch
from torchdata.nodes.base_node import BaseNode, T
from torchdata.nodes.samplers.utils import StopCriteria


class MultiDatasetWeightedSampler(BaseNode[T]):
    """A node that samples from multiple datasets with weights."""

    DATASET_NODE_STATES_KEY = "dataset_node_states"
    GENERATOR_STATE_KEY = "g_state"
    DATASETS_EXHAUSTED_KEY = "_datasets_exhausted"

    def __init__(
        self,
        source_nodes: Mapping[str, BaseNode[T]],
        weights: Dict[str, float],
        stopping_criterion: str = StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
    ) -> None:
        super().__init__()
        self.source_nodes = source_nodes
        self.weights = weights
        self.stopping_criterion = stopping_criterion
        self.dataset_names = list(self.source_nodes.keys())
        self._validate()

    def _validate(self) -> None:
        if self.stopping_criterion not in [
            StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            StopCriteria.ALL_DATASETS_EXHAUSTED,
            StopCriteria.FIRST_DATASET_EXHAUSTED,
        ]:
            raise ValueError(
                "stop_criteria must be one of: CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, FIRST_DATASET_EXHAUSTED, ALL_DATASETS_EXHAUSTED",
            )

        # Check if keys of source_nodes and weights are the same
        if set(self.dataset_names) != set(self.weights.keys()):
            raise ValueError(
                "For multi-dataset weighted sampling, keys of source_nodes and weights must be the same",
            )

        # Check if weights are 1d sequence
        if any(not isinstance(weight, float) for weight in self.weights.values()):
            raise ValueError(
                "For multi-dataset weighted sampling, weights must be a 1d sequence",
            )

        # Check if weights are non-negative
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError(
                "For multi-dataset weighted sampling, weights must be non-negative",
            )

        # Check if any weight is 0
        if any(weight == 0 for weight in self.weights.values()):
            raise ValueError(
                "For multi-dataset weighted sampling, weights must be non-zero",
            )

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)

        g = torch.Generator()
        g.manual_seed(0)  # TODO: incorporate seed from worker_info, epoch
        if initial_state is not None:
            self._weighted_sampler = _WeightedSampler(
                weights=self.weights,
                generator=g,
                initial_state=initial_state[self.GENERATOR_STATE_KEY],
            )
            self._datasets_exhausted = initial_state[self.DATASETS_EXHAUSTED_KEY]
            for k in self.dataset_names:
                self.source_nodes[k].reset(initial_state[self.DATASET_NODE_STATES_KEY][k])
        else:
            # Force a fresh iterator from all source nodes
            self._weighted_sampler = _WeightedSampler(
                weights=self.weights,
                generator=g,
            )
            self._datasets_exhausted = [False] * len(self.weights)
            for k in self.dataset_names:
                self.source_nodes[k].reset()

        self._ds_iters = [iter(self.source_nodes[k]) for k in self._weighted_sampler.names]

    def next(self) -> T:
        if all(self._datasets_exhausted):
            # Raise StopIteration if all datasets are exhausted
            raise StopIteration()

        idx, key = next(self._weighted_sampler)
        try:
            if self._datasets_exhausted[idx] and self.stopping_criterion == StopCriteria.ALL_DATASETS_EXHAUSTED:
                # Before fetching a new item check if idx corresponds to an already
                # exhaused dataset and StopCriteria is ALL_DATASETS_EXHAUSTED, move to next idx
                return self.next()
            item = next(self._ds_iters[idx])
        except StopIteration:
            # Mark the dataset as exhausted
            self._datasets_exhausted[idx] = True

            # Based on the stopping criterion, we may or may not raise StopIteration
            if self.stopping_criterion == StopCriteria.FIRST_DATASET_EXHAUSTED or (
                self.stopping_criterion
                in (
                    StopCriteria.ALL_DATASETS_EXHAUSTED,
                    StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
                )
                and all(self._datasets_exhausted)
            ):
                raise StopIteration()

            # If StopCriteria is ALL_DATASETS_EXHAUSTED, move to next idx
            if self.stopping_criterion == StopCriteria.ALL_DATASETS_EXHAUSTED:
                return self.next()

            # Reset the iterator and try again
            self._ds_iters[idx].reset()
            self._ds_iters[idx] = iter(self.source_nodes[key])
            item = next(self._ds_iters[idx])

        return item

    def get_state(self) -> Dict[str, Any]:
        return {
            self.GENERATOR_STATE_KEY: self._weighted_sampler.state_dict(),
            self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
            self.DATASET_NODE_STATES_KEY: {k: self.source_nodes[k].state_dict() for k in self.dataset_names},
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

    def state_dict(self):
        return {
            "g_state": self._g_snapshot,
            "offset": self._offset,
        }
