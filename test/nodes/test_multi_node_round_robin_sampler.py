# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import itertools
from enum import unique

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.batch import Batcher
from torchdata.nodes.prefetch import Prefetcher
from torchdata.nodes.samplers.multi_node_round_robin_sampler import (
    MultiNodeRoundRobinSampler,
)
from torchdata.nodes.samplers.stop_criteria import StopCriteria

from .utils import DummyIterableDataset, run_test_save_load_state


class TestMultiNodeRoundRobinSampler(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._num_samples = 1
        self._num_datasets = 3

    def get_equal_dataset(self, num_samples, num_datasets):
        datasets = {
            f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples, f"ds{i}"))
            for i in range(num_datasets)
        }
        return datasets

    def get_unequal_dataset(self, num_samples, num_datasets):
        datasets = {
            f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples + i, f"ds{i}"))
            for i in range(num_datasets)
        }
        return datasets

    def test_multi_node_round_robin_sampler_equal_dataset(self) -> None:
        datasets = self.get_equal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        batch_size = 3
        num_epochs = 1
        # each dataset has 1 sample, so the first and only epoch must be ['ds0', 'ds1', 'ds2']
        batcher = Batcher(sampler, batch_size=batch_size)
        for _ in range(num_epochs):
            results = next(batcher)
            self.assertGreater(len(results), 0)
            datasets_in_results = [result["name"] for result in results]
            dataset_counts_in_results = collections.Counter(datasets_in_results)
            for key in dataset_counts_in_results:
                self.assertEqual(dataset_counts_in_results[key], 1)

    def test_multi_node_round_robin_sampler_unequal_dataset(self) -> None:
        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.ALL_DATASETS_EXHAUSTED
        )
        batch_size = 3
        num_epochs = 2
        batcher = Batcher(sampler, batch_size=batch_size)
        # In this case, first epoch must be ['ds0', 'ds1', 'ds2'] and second epoch must be ['ds1', 'ds2', 'ds2']
        for epoch in range(num_epochs):
            results = next(batcher)
            self.assertGreater(len(results), 0)
            datasets_in_results = [result["name"] for result in results]
            dataset_counts_in_results = collections.Counter(datasets_in_results)
            if epoch == 0:
                self.assertEqual(len(dataset_counts_in_results), self._num_datasets)
                for key in dataset_counts_in_results:
                    self.assertEqual(dataset_counts_in_results[key], 1)
            elif epoch == 1:
                self.assertEqual(len(dataset_counts_in_results), self._num_datasets - 1)
                for key in dataset_counts_in_results:
                    if key == "ds0":
                        self.assertEqual(dataset_counts_in_results[key], 0)
                    if key == "ds1":
                        self.assertEqual(dataset_counts_in_results[key], 1)
                    else:
                        self.assertEqual(dataset_counts_in_results[key], 2)

    def test_get_state(self) -> None:
        datasets = self.get_equal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        state = sampler.get_state()
        self.assertIn("current_dataset_index", state)
        self.assertIn("datasets_exhausted", state)
        self.assertIn("dataset_node_states", state)

    def test_multi_node_round_robin_large_sample_size(self) -> None:
        num_samples = 1500
        num_datasets = 3
        datasets = self.get_equal_dataset(num_samples, num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.ALL_DATASETS_EXHAUSTED
        )
        prefetcher = Prefetcher(sampler, 3)
        run_test_save_load_state(self, prefetcher, 400)
