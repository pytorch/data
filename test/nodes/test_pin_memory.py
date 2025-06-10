# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from unittest import mock

import torch

from parameterized import parameterized

from torch.testing._internal.common_utils import TEST_CUDA, TestCase

from torchdata.nodes.batch import Batcher
from torchdata.nodes.map import Mapper
from torchdata.nodes.pin_memory import PinMemory
from torchdata.nodes.prefetch import Prefetcher

from .utils import Collate, IterInitError, MockSource, run_test_save_load_state, StatefulRangeNode


@unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
class TestPinMemory(TestCase):
    def test_pin_memory(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size)
        node = Mapper(node, Collate())
        node = PinMemory(node)
        root = Prefetcher(node, prefetch_factor=2)

        # 2 epochs
        for epoch in range(2):
            root.reset()
            results = list(root)
            self.assertEqual(len(results), 3, epoch)
            for i in range(3):
                for j in range(batch_size):
                    self.assertEqual(results[i]["step"][j], i * batch_size + j)
                    self.assertEqual(results[i]["test_tensor"][j], torch.tensor([i * batch_size + j]))
                    self.assertEqual(results[i]["test_str"][j], f"str_{i * batch_size + j}")

    def test_exception_handling(self):
        class PinMemoryFails:
            def pin_memory(self):
                raise ValueError("test exception")

        batch_size = 6
        src = MockSource(num_samples=20)
        node = Mapper(src, lambda x: dict(fail=PinMemoryFails(), **x))
        node = Batcher(node, batch_size=batch_size)
        node = Mapper(node, Collate())
        node = PinMemory(node)
        root = Prefetcher(node, prefetch_factor=2)

        with self.assertRaisesRegex(ValueError, "test exception"):
            list(root)

    def test_iter_init_error(self):
        node = IterInitError()
        node = PinMemory(node)
        root = Prefetcher(node, prefetch_factor=2)

        with self.assertRaisesRegex(ValueError, "Iter Init Error"):
            list(root)

    @parameterized.expand(itertools.product([0, 7, 33], [0, 1, 9]))
    def test_save_load_state_stateful(self, midpoint: int, snapshot_frequency: int):
        batch_size = 6
        n = 200
        node = StatefulRangeNode(n=n)
        node = Batcher(node, batch_size=batch_size, drop_last=False)
        node = Mapper(node, Collate())
        node = PinMemory(node, snapshot_frequency=snapshot_frequency)
        node = Prefetcher(node, prefetch_factor=8)

        run_test_save_load_state(self, node, midpoint)

    def test_explicit_shutdown(self):
        """Test that the explicit shutdown method properly shuts down the node."""
        mock_source = mock.MagicMock()
        mock_source.shutdown = mock.MagicMock()
        node = PinMemory(
            mock_source,
        )
        node.reset()
        # Mock the _shutdown method of the iterator
        with mock.patch.object(node._it, "_shutdown") as mock_shutdown:
            node.shutdown()
            mock_shutdown.assert_called_once()
            mock_source.shutdown.assert_called_once()
