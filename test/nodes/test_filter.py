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
from torchdata.nodes.filter import Filter

from .utils import MockSource, run_test_save_load_state, StatefulRangeNode


class TestFilter(TestCase):
    def _test_filter(self, num_workers, in_order, method):
        n = 100
        predicate = lambda x: x["test_tensor"] % 2 == 0  # Filter even numbers
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
            {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"}
            for i in range(n)
            if i % 2 == 0
        ]
        self.assertEqual(results, expected_results)

    def test_filter_inline(self):
        self._test_filter(num_workers=0, in_order=True, method="thread")

    def test_filter_parallel_threads(self):
        self._test_filter(num_workers=4, in_order=True, method="thread")

    def test_filter_parallel_process(self):
        self._test_filter(num_workers=4, in_order=True, method="process")

    @parameterized.expand(
        itertools.product(
            [0],  # , 7, 13],
            [True],  # TODO: define and fix in_order = False
            [0],  # , 1, 9],  # TODO: define and fix in_order = False
        )
    )
    def test_save_load_state_thread(
        self, midpoint: int, in_order: bool, snapshot_frequency: int
    ):
        method = "thread"
        n = 100
        predicate = lambda x: True
        src = StatefulRangeNode(n=n)

        node = Filter(
            source=src,
            predicate=predicate,
            num_workers=4,
            in_order=in_order,
            method=method,
            snapshot_frequency=snapshot_frequency,
        )
        node.reset()
        run_test_save_load_state(self, node, midpoint)
