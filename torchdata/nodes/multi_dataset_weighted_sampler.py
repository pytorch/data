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
    NUM_RAND_LIST_SKIP_KEY = "_num_rand_list_skip"
    DATASETS_EXHAUSTED_KEY = "_datasets_exhausted"

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
        self._num_yielded = 0
        self._num_rand_list_skip = 0
        self._datasets_exhausted = [False] * len(self.weights)

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        """Logic for iterating over multiple datasets with weights"""
        if initial_state is not None:  # TODO add more checks here
            # load state from initial_state
            print("[MultiDatasetWeightedSampler] Loading state from initial_state")
            dataset_node_state = initial_state[self.DATASET_NODE_STATES_KEY]
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
            self._num_rand_list_skip = initial_state[self.NUM_RAND_LIST_SKIP_KEY]
            self._datasets_exhausted = initial_state[self.DATASETS_EXHAUSTED_KEY]
            print(self._num_yielded, self._num_rand_list_skip, self._datasets_exhausted)
            # load state for each parent dataset node
            for dataset_key in self.data_nodes.keys():
                self.data_nodes[dataset_key].load_state_dict(dataset_node_state[dataset_key])

        # create iterators for each dataset node
        dataset_iterators = {
            dataset_idx: (
                iter(self.data_nodes[dataset_key])
            )  # doing iter(...) will reset the iterator if an initial_state is already loaded
            for dataset_idx, dataset_key in enumerate(self.data_nodes.keys())
        }
        # if false:
        # print(
        #     "[test state loaded correctly]:1 ", self.data_nodes["ds3"].get_state()
        # )  # SEEMS CORRECTLY LOADED, but dataset_iterators might have the wrong stuff | better annotate each dataset
        # print("[test state loaded correctly]:2 ", dataset_iterators[3].started())
        # print(
        #     "[test state loaded correctly]:3 ",
        #     self.data_nodes["ds3"]._get_iterator().started(),
        # )

        # aggregate state across parent nodes
        # skip some iterations based on loaded state

        # create a weighted sampler
        g = torch.Generator()
        g.manual_seed(0)  # TODO: incorporate seed from worker_info, epoch
        while all(self._datasets_exhausted) is False:
            # print("Inside while loop")
            weighted_sampler = torch.multinomial(
                torch.tensor(list(self.weights.values())),
                num_samples=self._ARBITRARY_LARGE_NUMBER,
                replacement=True,
                generator=g,
            ).tolist()

            # iterate over the weighted sampler
            for dataset_idx in weighted_sampler[self._num_yielded + self._num_rand_list_skip :]:
                # yield from the iterator for the dataset
                try:
                    if all(self._datasets_exhausted):
                        break
                    if self._datasets_exhausted[dataset_idx]:
                        self._num_rand_list_skip += 1
                        continue  # skip this dataset and move on to the next one
                    # yield from dataset_iterators[
                    #     dataset_idx
                    # ]  # not correct, as it will keep looping over one dataset
                    # yield next(
                    #     dataset_iterators[dataset_idx].iterator(None)
                    # )  # not recommended, plus not correct
                    # if fast_forward_counter > 0:
                    #     next(dataset_iterators[dataset_idx])
                    #     fast_forward_counter -= 1
                    #     continue
                    self._num_yielded += 1
                    yield next(dataset_iterators[dataset_idx])
                except (StopIteration):  # TODO: handle the case where all datasets are exhausted | StopIteration
                    # mark the dataset as exhausted
                    print(dataset_idx, " has exhaused")
                    self._datasets_exhausted[dataset_idx] = True
                    self._num_rand_list_skip += 1
                    self._num_yielded -= 1
        # print("All datasets have exhausted")

    def get_state(self) -> Dict[str, Any]:
        # TODO: add more keys here
        return {
            self.NUM_YIELDED_KEY: self._num_yielded,
            self.NUM_RAND_LIST_SKIP_KEY: self._num_rand_list_skip,
            self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
            self.DATASET_NODE_STATES_KEY: {
                dataset_key: self.data_nodes[dataset_key].state_dict() for dataset_key in self.data_nodes.keys()
            },
        }
