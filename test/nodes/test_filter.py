# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from typing import List

import torch

from parameterized import parameterized
from torch.testing._internal.common_utils import IS_WINDOWS, TEST_CUDA, TestCase

from torchdata.nodes.base_node import BaseNode
from torchdata.nodes.batch import Batcher
from torchdata.nodes.filter import Filter
from torchdata.nodes.samplers.multi_node_weighted_sampler import MultiNodeWeightedSampler

from .utils import MockSource, run_test_save_load_state, StatefulRangeNode


class TestFilter(TestCase):
    def _test_filter(self, num_workers, in_order, method):
        n = 100

        def predicate(x):
            return x["test_tensor"] % 2 == 0

        src = MockSource(num_samples=n)
        node = Filter(
            source=src,
            predicate=predicate,
            num_workers=num_workers,
            in_order=in_order,
            method=method,
        )

        results: List[int] = []
        for item in node:
            results.append(item)

        expected_results = [
            {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"} for i in range(n) if i % 2 == 0
        ]
        self.assertEqual(results, expected_results)

    def test_filter_inline(self):
        self._test_filter(num_workers=0, in_order=True, method="thread")

    def test_filter_parallel_threads(self):
        self._test_filter(num_workers=4, in_order=True, method="thread")

    def test_filter_parallel_process(self):
        self._test_filter(num_workers=4, in_order=True, method="process")

    @parameterized.expand([100, 200, 300])
    def test_filter_batcher(self, n):
        src = StatefulRangeNode(n=n)
        node = Batcher(src, batch_size=2)

        def predicate(x):
            return (x[0]["i"] + x[1]["i"]) % 3 == 0

        node = Filter(node, predicate, num_workers=2)
        results = list(node)
        self.assertEqual(len(results), n // 6)

    @parameterized.expand(
        itertools.product(
            [10, 20, 40],
            [True],  # TODO: define and fix in_order = False
            [1, 2, 4],
        )
    )
    def test_save_load_state_thread(self, midpoint: int, in_order: bool, snapshot_frequency: int):
        method = "thread"
        n = 100

        def predicate(x):
            return x["i"] % 2 == 0

        src = StatefulRangeNode(n=n)

        node = Filter(
            source=src,
            predicate=predicate,
            num_workers=1,
            in_order=in_order,
            method=method,
            snapshot_frequency=snapshot_frequency,
        )
        node.reset()
        run_test_save_load_state(self, node, midpoint)
