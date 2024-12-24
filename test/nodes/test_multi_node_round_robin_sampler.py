# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import itertools
import math

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase

from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.batch import Batcher
from torchdata.nodes.loader import Loader
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
        """Returns a dictionary of datasets with the same number of samples"""
        datasets = {
            f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples, f"ds{i}"))
            for i in range(num_datasets)
        }
        return datasets

    def get_unequal_dataset(self, num_samples, num_datasets):
        """Returns a dictionary of datasets with the different number of samples.
        For example if num_samples = 1 and num_datasets = 3, the datasets will have 1, 2, 3 samples, respectively.
        datasets = {"ds0":[0], "ds1":[0, 1], "ds2":[0, 1, 2]}
        """
        datasets = {
            f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples + i, f"ds{i}"))
            for i in range(num_datasets)
        }
        return datasets

    def test_empty_datasets(self) -> None:
        datasets = self.get_equal_dataset(0, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        for item in sampler:
            self.fail("Expected no batches as each dataset is empty.")

    def test_empty_datasets_batched(self) -> None:
        datasets = self.get_equal_dataset(0, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        batch_size = 3
        batcher = Batcher(sampler, batch_size=batch_size)
        for batch in batcher:
            self.fail("Expected no batches as each dataset is empty.")

    @parameterized.expand([4, 8, 16])
    def test_single_dataset(self, num_samples: int) -> None:
        datasets = self.get_equal_dataset(num_samples, 1)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        for num_sample, item in enumerate(sampler):
            pass
        self.assertEqual(num_sample + 1, num_samples)

    @parameterized.expand([4, 8, 16])
    def test_single_dataset_batched(self, num_samples: int) -> None:
        datasets = self.get_equal_dataset(num_samples, 1)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        batch_size = 4
        batcher = Batcher(sampler, batch_size=batch_size)
        for batch_number, batch in enumerate(batcher):
            self.assertGreater(len(batch), 0)
            self.assertEqual(len(batch), batch_size)
        self.assertEqual(batch_number + 1, num_samples // batch_size)

    @parameterized.expand(
        itertools.product(
            [8, 16, 32],
            [True, False],
        )
    )
    def test_single_dataset_drop_last_batched(
        self, num_samples: int, drop_last: bool
    ) -> None:
        datasets = self.get_equal_dataset(num_samples, 1)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        batch_size = 5
        batcher = Batcher(sampler, batch_size=batch_size, drop_last=drop_last)
        num_batches = 0
        for batch_number, batch in enumerate(batcher):
            num_batches += 1
            self.assertGreater(len(batch), 0)
            if drop_last:
                self.assertEqual(len(batch), batch_size)
        if drop_last:
            self.assertEqual(num_batches, math.ceil(num_samples / batch_size) - 1)
        else:
            self.assertEqual(num_batches, math.ceil(num_samples / batch_size))

    def test_stop_criteria_all_datasets_exhausted(self) -> None:
        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        total_items = sum(
            range(self._num_samples, self._num_samples + self._num_datasets)
        )
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.ALL_DATASETS_EXHAUSTED
        )
        for num_sample, item in enumerate(sampler):
            pass
        self.assertEqual(num_sample + 1, total_items)

    def test_stop_criteria_all_datasets_exhausted_batched(self) -> None:
        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        total_items = sum(
            range(self._num_samples, self._num_samples + self._num_datasets)
        )
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.ALL_DATASETS_EXHAUSTED
        )
        batch_size = 3
        batcher = Batcher(sampler, batch_size=batch_size, drop_last=True)
        num_batches = 0
        for batch in batcher:
            num_batches += 1
        self.assertEqual(num_batches, total_items // batch_size)

    def test_stop_criteria_first_dataset_exhausted(self) -> None:
        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        for num_sample, item in enumerate(sampler):
            pass
        self.assertEqual(num_sample + 1, self._num_datasets)

    def test_stop_criteria_first_dataset_exhausted_batched(self) -> None:
        num_samples = 4
        datasets = self.get_unequal_dataset(
            num_samples, self._num_datasets
        )  # first dataset has 4 samples, second has 5, and third has 6
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        batch_size = 2
        batcher = Batcher(sampler, batch_size=batch_size)
        num_batches = 0
        for batch in batcher:
            num_batches += 1
        self.assertEqual(num_batches, num_samples * self._num_datasets // batch_size)

    def test_stop_criteria_cycle_until_all_datasets_exhausted(self) -> None:
        num_samples = 4
        datasets = self.get_unequal_dataset(num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
        )
        for num_sample, item in enumerate(sampler):
            pass
        self.assertEqual(
            num_sample + 1, self._num_datasets * (num_samples + self._num_datasets - 1)
        )

    def test_stop_criteria_cycle_until_all_datasets_exhausted_batched(self) -> None:
        num_samples = 4
        datasets = self.get_unequal_dataset(num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
        )
        batch_size = 3
        batcher = Batcher(sampler, batch_size=batch_size)
        num_batches = 0
        for batch in batcher:
            num_batches += 1
        self.assertEqual(num_batches, (num_samples + self._num_datasets - 1))

    def test_multi_node_round_robin_sampler_equal_dataset_batched(self) -> None:
        datasets = self.get_equal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        batch_size = 3
        # each dataset has 1 sample, so the first and only batch must be ['ds0', 'ds1', 'ds2']
        batcher = Batcher(sampler, batch_size=batch_size)
        for batch in batcher:
            self.assertGreater(len(batch), 0)
            datasets_in_batch = [result["name"] for result in batch]
            dataset_counts_in_batch = collections.Counter(datasets_in_batch)
            for key in dataset_counts_in_batch:
                self.assertEqual(dataset_counts_in_batch[key], 1)

    def test_multi_node_round_robin_sampler_unequal_dataset_batched(self) -> None:

        # First we test with StopCriteria.ALL_DATASETS_EXHAUSTED
        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.ALL_DATASETS_EXHAUSTED
        )
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
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
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
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
        )
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

    def test_unequal_batch_size(self) -> None:
        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)

        node1 = Batcher(datasets["ds0"], 1)
        node2 = Batcher(datasets["ds1"], 2)
        node3 = Batcher(datasets["ds2"], 3)

        node = MultiNodeRoundRobinSampler(
            {"node1": node1, "node2": node2, "node3": node3},
            StopCriteria.ALL_DATASETS_EXHAUSTED,
        )

        loader = Loader(node, 1)
        for batch_num, batch in enumerate(loader):
            self.assertGreater(len(batch), 0)
            datasets_in_batch = list({result["name"] for result in batch})
            self.assertEqual(len(datasets_in_batch), 1)
            self.assertEqual(datasets_in_batch[0], f"ds{batch_num}")
            self.assertEqual(len(batch), batch_num + 1)

    def test_get_state(self) -> None:
        datasets = self.get_equal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        state = sampler.get_state()
        self.assertIn("current_dataset_index", state)
        self.assertIn("datasets_exhausted", state)
        self.assertIn("dataset_node_states", state)

    @parameterized.expand(
        itertools.product(
            [100, 500, 1200],
            [
                StopCriteria.ALL_DATASETS_EXHAUSTED,
                StopCriteria.FIRST_DATASET_EXHAUSTED,
                StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            ],
        )
    )
    def test_save_load_state(self, midpoint: int, stop_criteria: str) -> None:
        num_samples = 1500
        num_datasets = 3
        datasets = self.get_equal_dataset(num_samples, num_datasets)
        sampler = MultiNodeRoundRobinSampler(datasets, stop_criteria)
        prefetcher = Prefetcher(sampler, 3)
        run_test_save_load_state(self, prefetcher, midpoint)

        datasets = self.get_unequal_dataset(num_samples, num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.ALL_DATASETS_EXHAUSTED
        )
        prefetcher = Prefetcher(sampler, 3)
        run_test_save_load_state(self, prefetcher, 400)

    def test_multiple_epochs_batched(self) -> None:
        num_epochs = 2

        datasets = self.get_unequal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
        )
        batch_size = 3
        batcher = Batcher(sampler, batch_size=batch_size)
        loader = Loader(batcher)
        results = {}

        for epoch in range(num_epochs):
            results[epoch] = []
            for batch in loader:
                results[epoch].append(batch)
        self.assertEqual(results[0], results[1])

    def test_get_state_after_reset_batched(self) -> None:
        datasets = self.get_equal_dataset(self._num_samples, self._num_datasets)
        sampler = MultiNodeRoundRobinSampler(
            datasets, StopCriteria.FIRST_DATASET_EXHAUSTED
        )
        batch_size = 3
        batcher = Batcher(sampler, batch_size=batch_size)
        next(batcher)
        state_before_reset = sampler.get_state()
        sampler.reset()
        state_after_reset = sampler.get_state()
        self.assertNotEqual(state_before_reset, state_after_reset)
