# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.batch import Batcher
from torchdata.nodes.loader import Loader
from torchdata.nodes.prefetch import Prefetcher

from .utils import IterInitError, MockSource, run_test_save_load_state, StatefulRangeNode


class TestPrefetcher(TestCase):
    def test_prefetcher(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size, drop_last=True)
        root = Prefetcher(node, prefetch_factor=2)

        # Test multi epoch shutdown and restart
        for _ in range(2):
            root.reset()
            results = list(root)
            self.assertEqual(len(results), 3)
            for i in range(3):
                for j in range(batch_size):
                    self.assertEqual(results[i][j]["step"], i * batch_size + j)
                    self.assertEqual(results[i][j]["test_tensor"], torch.tensor([i * batch_size + j]))
                    self.assertEqual(results[i][j]["test_str"], f"str_{i * batch_size + j}")

    def test_iter_init_error(self):
        node = IterInitError()
        root = Prefetcher(node, prefetch_factor=2)

        with self.assertRaisesRegex(ValueError, "Iter Init Error"):
            list(root)

    @parameterized.expand(itertools.product([0, 7, 32], [0, 1, 9]))
    def test_save_load_state_stateful(self, midpoint: int, snapshot_frequency: int):
        batch_size = 6
        n = 200
        src = MockSource(num_samples=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = Prefetcher(node, prefetch_factor=8, snapshot_frequency=snapshot_frequency)
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand([0, 1, 9])
    def test_initial_snapshot(self, snapshot_frequency: int):
        n = 200
        src = StatefulRangeNode(n=n)
        node = Prefetcher(src, prefetch_factor=8, snapshot_frequency=snapshot_frequency)
        initial_state = node.state_dict()

        exp = []
        for _ in range(10):
            exp.append(next(node))
        mid_state = node.state_dict()
        exp2 = list(node)

        node.reset(initial_state)
        result = []
        for _ in range(10):
            result.append(next(node))
        node.reset(mid_state)
        result2 = list(node)

        self.assertEqual(exp, result)
        self.assertEqual(exp2, result2)

        # Test with Loader
        loader = Loader(node)
        initial_state = loader.state_dict()
        it = iter(loader)

        exp = []
        for _ in range(10):
            exp.append(next(it))
        mid_state = loader.state_dict()
        exp2 = list(it)

        loader.load_state_dict(initial_state)
        it = iter(loader)
        result = []
        for _ in range(10):
            result.append(next(it))
        mid_state = loader.load_state_dict(mid_state)
        result2 = list(it)

        self.assertEqual(exp, result)
        self.assertEqual(exp2, result2)
