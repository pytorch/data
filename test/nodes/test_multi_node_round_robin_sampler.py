# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import itertools
from enum import unique

import torch

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.batch import Batcher
from torchdata.nodes.prefetch import Prefetcher
from torchdata.nodes.samplers.multi_node_round_robin_sampler import MultiNodeRoundRobinSampler
from torchdata.nodes.samplers.stop_criteria import StopCriteria

from .utils import DummyIterableDataset, run_test_save_load_state


class TestMultiNodeRoundRobinSampler(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._num_samples = 1
        self._num_datasets = 3

    def get_equal_dataset(self, num_samples, num_datasets):
        """Returns a dictionary of datasets with the same number of samples"""
        datasets = {f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples, f"ds{i}")) for i in range(num_datasets)}
        return datasets

    def get_unequal_dataset(self, num_samples, num_datasets):
        """Returns a dictionary of datasets with the different number of samples.
        For example if num_samples = 1 and num_datasets = 3, the datasets will have 1, 2, 3 samples, respectively.
        datasets = {"ds0":[0], "ds1":[0, 1], "ds2":[0, 1, 2]}
        """
        datasets = {
            f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples + i, f"ds{i}")) for i in range(num_datasets)
        }
        return datasets

    def test_multi_node_round_robin_sampler_equal_dataset(self) -> None:
        datasets = self.get_equal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(datasets, StopCriteria.FIRST_DATASET_EXHAUSTED)
        batch_size = 3
        # each dataset has 1 sample, so the first and only batch must be ['ds0', 'ds1', 'ds2']
        batcher = Batcher(sampler, batch_size=batch_size)
        for batch in batcher:
            self.assertGreater(len(batch), 0)
            datasets_in_batch = [result["name"] for result in batch]
            dataset_counts_in_batch = collections.Counter(datasets_in_batch)
            for key in dataset_counts_in_batch:
                self.assertEqual(dataset_counts_in_batch[key], 1)

    def test_multi_node_round_robin_sampler_unequal_dataset(self) -> None:

        # First we test with StopCriteria.ALL_DATASETS_EXHAUSTED
        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(datasets, StopCriteria.ALL_DATASETS_EXHAUSTED)
        batch_size = 3
        batcher = Batcher(sampler, batch_size=batch_size)
        # In this case, first batch must be ['ds0', 'ds1', 'ds2'] and second batch must be ['ds1', 'ds2', 'ds2']
        for batch_number, batch in enumerate(batcher):
            self.assertGreater(len(batch), 0)
            datasets_in_batch = [result["name"] for result in batch]
            dataset_counts_in_batch = collections.Counter(datasets_in_batch)
            if batch_number == 0:
                self.assertEqual(len(dataset_counts_in_batch), self._num_datasets)
                for key in dataset_counts_in_batch:
                    self.assertEqual(dataset_counts_in_batch[key], 1)
            elif batch_number == 1:
                self.assertEqual(len(dataset_counts_in_batch), self._num_datasets - 1)
                for key in dataset_counts_in_batch:
                    if key == "ds0":
                        self.assertEqual(dataset_counts_in_batch[key], 0)
                    if key == "ds1":
                        self.assertEqual(dataset_counts_in_batch[key], 1)
                    else:
                        self.assertEqual(dataset_counts_in_batch[key], 2)
        self.assertEqual(batch_number, 1)

        # Now we test with StopCriteria.FIRST_DATASET_EXHAUSTED
        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(datasets, StopCriteria.FIRST_DATASET_EXHAUSTED)
        batch_size = 3
        batcher = Batcher(sampler, batch_size=batch_size)
        # In this case, there will just be one batch ['ds0', 'ds1', 'ds2']
        for batch_number, batch in enumerate(batcher):
            self.assertGreater(len(batch), 0)
            datasets_in_batch = [result["name"] for result in batch]
            dataset_counts_in_batch = collections.Counter(datasets_in_batch)
            if batch_number == 0:
                self.assertEqual(len(dataset_counts_in_batch), self._num_datasets)
                for key in dataset_counts_in_batch:
                    self.assertEqual(dataset_counts_in_batch[key], 1)
        self.assertEqual(batch_number, 0)

        # Finally we test with StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED

        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(datasets, StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED)
        batch_size = 3
        batcher = Batcher(sampler, batch_size=batch_size)
        # In this case, we will have a total of 3 batches.
        # Each batch will be ['ds0', 'ds1', 'ds2'], but since ds0 has only 1 sample, it will yield the same item in all the batches
        # ds1 has 2 samples, so its item in the first and third batch will be the same
        # ds2 has 3 samples, so its items in all the three batches will be different
        for batch_number, batch in enumerate(batcher):
            self.assertGreater(len(batch), 0)
            datasets_in_batch = [result["name"] for result in batch]
            dataset_counts_in_batch = collections.Counter(datasets_in_batch)

            self.assertEqual(len(dataset_counts_in_batch), self._num_datasets)
            for key in dataset_counts_in_batch:
                self.assertEqual(dataset_counts_in_batch[key], 1)

            self.assertEqual(batch[0]["name"], "ds0")
            self.assertEqual(batch[1]["name"], "ds1")
            self.assertEqual(batch[2]["name"], "ds2")
            self.assertEqual(batch[0]["test_tensor"], torch.tensor([0]))
            if batch_number == 0 or batch_number == 2:
                self.assertEqual(batch[1]["test_tensor"], torch.tensor([0]))
            else:
                self.assertEqual(batch[1]["test_tensor"], torch.tensor([1]))

            if batch_number == 0:
                self.assertEqual(batch[2]["test_tensor"], torch.tensor([0]))
            elif batch_number == 1:
                self.assertEqual(batch[2]["test_tensor"], torch.tensor([1]))
            else:
                self.assertEqual(batch[2]["test_tensor"], torch.tensor([2]))

        self.assertEqual(batch_number, 2)

    def test_get_state(self) -> None:
        datasets = self.get_equal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(datasets, StopCriteria.FIRST_DATASET_EXHAUSTED)
        state = sampler.get_state()
        self.assertIn("current_dataset_index", state)
        self.assertIn("datasets_exhausted", state)
        self.assertIn("dataset_node_states", state)

    def test_save_load_state(self) -> None:
        num_samples = 1500
        num_datasets = 3
        datasets = self.get_equal_dataset(num_samples, num_datasets)
        sampler = MultiNodeRoundRobinSampler(datasets, StopCriteria.ALL_DATASETS_EXHAUSTED)
        prefetcher = Prefetcher(sampler, 3)
        run_test_save_load_state(self, prefetcher, 400)

        datasets = self.get_unequal_dataset(num_samples, num_datasets)
        sampler = MultiNodeRoundRobinSampler(datasets, StopCriteria.ALL_DATASETS_EXHAUSTED)
        prefetcher = Prefetcher(sampler, 3)
        run_test_save_load_state(self, prefetcher, 400)
