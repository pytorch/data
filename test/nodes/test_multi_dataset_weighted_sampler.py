# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import MultiDatasetWeightedSampler
from torchdata.nodes.adapters import IterableWrapper

from .utils import DummyIterableDataset, run_test_save_load_state


class TestMultiDatasetWeightedSampler(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.datasets = {f"ds{i}": IterableWrapper(DummyIterableDataset(10, f"ds{i}")) for i in range(4)}
        self.weights = {f"ds{i}": 0.1 * (i + 1) for i in range(10)}
        self.weighted_sampler_node = MultiDatasetWeightedSampler(self.datasets, self.weights)

    def test_multi_dataset_weighted_sampler(self) -> None:
        # TODO: add test for Multi dataset weighted sampler functionality
        pass

    @parameterized.expand([2, 5])
    def test_save_load_state_stateful(self, midpoint: int):
        run_test_save_load_state(self, self.weighted_sampler_node, midpoint)
