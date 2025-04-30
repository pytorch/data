# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import Header
from torchdata.nodes.adapters import IterableWrapper

from .utils import MockSource, run_test_save_load_state, StatefulRangeNode


class TestHeader(TestCase):
    def test_header_basic(self) -> None:
        # Test with a simple range
        source = IterableWrapper(range(10))
        node = Header(source, n=5)

        results = list(node)
        self.assertEqual(results, [0, 1, 2, 3, 4])

        # Verify counter
        self.assertEqual(node._num_yielded, 5)

        # Test with n larger than source
        source = IterableWrapper(range(3))
        node = Header(source, n=10)

        results = list(node)
        self.assertEqual(results, [0, 1, 2])

        # Verify counter with n larger than source
        self.assertEqual(node._num_yielded, 3)

        # Test with n=0 (should yield nothing)
        source = IterableWrapper(range(10))
        node = Header(source, n=0)

        results = list(node)
        self.assertEqual(results, [])

        # Verify counter with n=0
        self.assertEqual(node._num_yielded, 0)

    def test_header_with_mock_source(self) -> None:
        num_samples = 20
        source = MockSource(num_samples=num_samples)
        node = Header(source, n=7)  # Limit to first 7 items

        # Test multi epoch
        for _ in range(2):
            node.reset()
            results = list(node)
            self.assertEqual(len(results), 7)

            # Verify counter after each epoch
            self.assertEqual(node._num_yielded, 7)

            for i, result in enumerate(results):
                expected_step = i
                self.assertEqual(result["step"], expected_step)
                self.assertEqual(result["test_tensor"].item(), expected_step)
                self.assertEqual(result["test_str"], f"str_{expected_step}")

    def test_header_empty_source(self) -> None:
        source = IterableWrapper([])
        node = Header(source, n=5)

        results = list(node)
        self.assertEqual(results, [])

        # Verify counter with empty source
        self.assertEqual(node._num_yielded, 0)

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state(self, midpoint: int) -> None:
        n = 50
        source = StatefulRangeNode(n=n)
        node = Header(source, n=20)  # Limit to first 20 items
        run_test_save_load_state(self, node, midpoint)

    def test_header_reset_state(self) -> None:
        source = IterableWrapper(range(10))
        node = Header(source, n=5)

        # Consume first two items
        self.assertEqual(next(node), 0)
        self.assertEqual(next(node), 1)

        # Check counter after consuming two items
        self.assertEqual(node._num_yielded, 2)

        # Get state and reset
        state = node.state_dict()
        node.reset(state)

        # Counter should be preserved after reset with state
        self.assertEqual(node._num_yielded, 2)

        # Should continue from where we left off
        self.assertEqual(next(node), 2)
        self.assertEqual(next(node), 3)
        self.assertEqual(next(node), 4)

        # Counter should be updated after consuming more items
        self.assertEqual(node._num_yielded, 5)

        # Should raise StopIteration after all items are consumed
        with self.assertRaises(StopIteration):
            next(node)

    def test_counter_reset(self) -> None:
        # Test that counter is properly reset
        source = IterableWrapper(range(10))
        node = Header(source, n=5)

        # Consume all items
        list(node)

        # Verify counter after first pass
        self.assertEqual(node._num_yielded, 5)

        # Reset without state
        node.reset()

        # Counter should be reset to 0
        self.assertEqual(node._num_yielded, 0)

        # Consume some items
        next(node)  # 0
        next(node)  # 1

        # Verify counter after partial consumption
        self.assertEqual(node._num_yielded, 2)

    def test_invalid_input(self) -> None:
        # Test with negative n
        source = IterableWrapper(range(10))
        with self.assertRaises(ValueError):
            Header(source, n=-1)
