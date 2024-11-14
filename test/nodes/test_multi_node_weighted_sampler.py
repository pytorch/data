# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper

from torchdata.nodes.samplers.multi_node_weighted_sampler import MultiNodeWeightedSampler
from torchdata.nodes.samplers.utils import StopCriteria

from .utils import DummyIterableDataset, run_test_save_load_state


class TestMultiNodeWeightedSampler(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._num_samples = 2
        self._num_datasets = 4
        self._weights_fn = lambda i: 0.1 * (i + 1)

        self.datasets = {
            f"ds{i}": IterableWrapper(DummyIterableDataset(self._num_samples, f"ds{i}"))
            for i in range(self._num_datasets)
        }
        self.weights = {f"ds{i}": self._weights_fn(i) for i in range(self._num_datasets)}
        self.weighted_sampler_node = MultiNodeWeightedSampler(self.datasets, self.weights)

    def test_multi_dataset_weighted_sampler_weight_sampler_keys_mismatch(self) -> None:
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

    def test_multi_dataset_weighted_batch_sampler_invalid_weights_tensor_shape(
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

    def test_multi_dataset_weighted_batch_sampler_negative_weights(
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

    def test_multi_dataset_weighted_batch_sampler_zero_weights(
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

    def test_multi_dataset_weighted_sampler_first_exhausted(self) -> None:
        mixer = MultiNodeWeightedSampler(
            self.datasets,
            self.weights,
            stop_criteria=StopCriteria.FIRST_DATASET_EXHAUSTED,
        )
        results = list(mixer)
        datasets_in_results = {result["name"] for result in results}

        self.assertEqual("ds3" in datasets_in_results, True)  # ds3 has maximum weight
        self.assertEqual("ds0" in datasets_in_results, False)  # ds0 has minimum weight

    def test_multi_dataset_weighted_sampler_all_dataset_exhausted(self) -> None:
        mixer = MultiNodeWeightedSampler(
            self.datasets,
            self.weights,
            stop_criteria=StopCriteria.ALL_DATASETS_EXHAUSTED,
        )
        results = list(mixer)
        datasets_in_results = [result["name"] for result in results]

        # check each dataset appears exactly _num_samples times,
        # each dataset has _num_samples samples
        self.assertEqual(
            [datasets_in_results.count(f"ds{i}") for i in range(self._num_datasets)],
            [self._num_samples] * self._num_datasets,
        )

        # check that all datasets are exhausted
        self.assertEqual(sorted(set(datasets_in_results)), ["ds0", "ds1", "ds2", "ds3"])

    def test_multi_dataset_weighted_sampler_cycle_until_all_exhausted(self) -> None:
        mixer = self.weighted_sampler_node
        results = list(mixer)
        datasets_in_results = {result["name"] for result in results}
        self.assertEqual(sorted(datasets_in_results), ["ds0", "ds1", "ds2", "ds3"])

    @parameterized.expand([2, 5, 7])
    def test_save_load_state_stateful(self, midpoint: int):
        run_test_save_load_state(self, self.weighted_sampler_node, midpoint)
