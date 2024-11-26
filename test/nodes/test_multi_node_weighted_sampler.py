# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from enum import unique

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.prefetch import Prefetcher

from torchdata.nodes.samplers.multi_node_weighted_sampler import MultiNodeWeightedSampler
from torchdata.nodes.samplers.stop_criteria import StopCriteria

from .utils import DummyIterableDataset, run_test_save_load_state


class TestMultiNodeWeightedSampler(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._num_samples = 10
        self._num_datasets = 4
        self._weights_fn = lambda i: 0.1 * (i + 1)
        self._num_epochs = 3

        self.datasets = {
            f"ds{i}": IterableWrapper(DummyIterableDataset(self._num_samples, f"ds{i}"))
            for i in range(self._num_datasets)
        }
        self.weights = {f"ds{i}": self._weights_fn(i) for i in range(self._num_datasets)}

    def test_torchdata_nodes_imports(self) -> None:
        try:
            from torchdata.nodes import MultiNodeWeightedSampler, StopCriteria  # noqa
        except ImportError:
            self.fail("MultiNodeWeightedSampler or StopCriteria failed to import")

    def _setup_multi_node_wighted_sampler(self, num_samples, num_datasets, weights_fn, stop_criteria) -> Prefetcher:

        datasets = {f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples, f"ds{i}")) for i in range(num_datasets)}
        weights = {f"ds{i}": weights_fn(i) for i in range(num_datasets)}
        node = MultiNodeWeightedSampler(datasets, weights, stop_criteria)
        return Prefetcher(node, prefetch_factor=3)

    def test_multi_node_weighted_sampler_weight_sampler_keys_mismatch(self) -> None:
        """
        Validation should fail if the keys of source_nodes and weights are not the same
        """
        with self.assertRaisesRegex(
            ValueError,
            "keys of source_nodes and weights must be the same",
        ):
            MultiNodeWeightedSampler(
                self.datasets,
                {f"dummy{i}": self._weights_fn(i) for i in range(self._num_datasets)},
            )

    def test_multi_node_weighted_batch_sampler_invalid_weights_tensor_shape(
        self,
    ) -> None:
        """
        Validation should fail if the shape of the weights tensor is invalid
        """
        with self.assertRaisesRegex(ValueError, " weights must be a 1d sequence, non-negative, and non-zero"):
            MultiNodeWeightedSampler(
                self.datasets,
                weights={f"ds{i}": [[1.0]] for i in range(self._num_datasets)},
            )

    def test_multi_node_weighted_batch_sampler_negative_weights(
        self,
    ) -> None:
        """
        Validation should fail if the value of the weights tensor is invalid
        """
        with self.assertRaisesRegex(ValueError, " weights must be a 1d sequence, non-negative, and non-zero"):
            MultiNodeWeightedSampler(
                self.datasets,
                weights={f"ds{i}": -1 for i in range(self._num_datasets)},
            )

    def test_multi_node_weighted_batch_sampler_zero_weights(
        self,
    ) -> None:
        """
        Validation should fail if the value of the weights tensor is invalid
        """
        with self.assertRaisesRegex(ValueError, " weights must be a 1d sequence, non-negative, and non-zero"):
            MultiNodeWeightedSampler(
                self.datasets,
                weights={f"ds{i}": 10 * i for i in range(self._num_datasets)},
            )

    def test_multi_node_weighted_sampler_first_exhausted(self) -> None:
        mixer = self._setup_multi_node_wighted_sampler(
            self._num_samples,
            self._num_datasets,
            self._weights_fn,
            stop_criteria=StopCriteria.FIRST_DATASET_EXHAUSTED,
        )

        for _ in range(self._num_epochs):
            results = list(mixer)

            datasets_in_results = [result["name"] for result in results]
            dataset_counts_in_results = [datasets_in_results.count(f"ds{i}") for i in range(self._num_datasets)]

            # Check max item count for dataset is exactly _num_samples
            self.assertEqual(max(dataset_counts_in_results), self._num_samples)

            # Check only one dataset has been exhausted
            self.assertEqual(dataset_counts_in_results.count(self._num_samples), 1)
            mixer.reset()

    def test_multi_node_weighted_sampler_all_dataset_exhausted(self) -> None:
        mixer = self._setup_multi_node_wighted_sampler(
            self._num_samples,
            self._num_datasets,
            self._weights_fn,
            stop_criteria=StopCriteria.ALL_DATASETS_EXHAUSTED,
        )

        for _ in range(self._num_epochs):
            results = list(mixer)
            datasets_in_results = [result["name"] for result in results]
            dataset_counts_in_results = [datasets_in_results.count(f"ds{i}") for i in range(self._num_datasets)]

            # check each dataset appears exactly _num_samples times,
            # each dataset has _num_samples samples
            self.assertEqual(
                dataset_counts_in_results,
                [self._num_samples] * self._num_datasets,
            )

            # check that all datasets are exhausted
            self.assertEqual(sorted(set(datasets_in_results)), ["ds0", "ds1", "ds2", "ds3"])
            mixer.reset()

    def test_multi_node_weighted_sampler_cycle_until_all_exhausted(self) -> None:
        mixer = self._setup_multi_node_wighted_sampler(
            self._num_samples,
            self._num_datasets,
            self._weights_fn,
            stop_criteria=StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
        )

        for _ in range(self._num_epochs):
            results = list(mixer)
            datasets_in_results = {result["name"] for result in results}

            # check that all datasets are exhausted
            self.assertEqual(sorted(datasets_in_results), ["ds0", "ds1", "ds2", "ds3"])
            mixer.reset()

    @parameterized.expand(
        itertools.product(
            [1, 4, 7],
            [
                StopCriteria.ALL_DATASETS_EXHAUSTED,
                StopCriteria.FIRST_DATASET_EXHAUSTED,
                StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
            ],
        )
    )
    def test_save_load_state_stateful(self, midpoint: int, stop_criteria: str):
        mixer = MultiNodeWeightedSampler(self.datasets, self.weights, stop_criteria)
        run_test_save_load_state(self, mixer, midpoint)

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
    def test_multi_node_weighted_large_sample_size(self, midpoint, stop_criteria) -> None:
        num_samples = 1500
        num_datasets = 5

        mixer = self._setup_multi_node_wighted_sampler(
            num_samples,
            num_datasets,
            self._weights_fn,
            stop_criteria,
        )
        run_test_save_load_state(self, mixer, midpoint)

    @parameterized.expand([(1, 8), (8, 32)])
    def test_multi_node_weighted_batch_sampler_set_rank_world_size(self, rank, world_size):
        mixer = MultiNodeWeightedSampler(self.datasets, self.weights, rank=rank, world_size=world_size)
        self.assertEqual(mixer.rank, rank)
        self.assertEqual(mixer.world_size, world_size)

    def test_multi_node_weighted_batch_sampler_results_for_ranks(self):
        world_size = 8
        global_results = []
        for rank in range(world_size):
            mixer = MultiNodeWeightedSampler(self.datasets, self.weights, rank=rank, world_size=world_size)
            results = list(mixer)
            global_results.append(results)

        unique_results = []
        for results in global_results:
            if results not in unique_results:
                unique_results.append(results)
        self.assertEqual(unique_results, global_results)
