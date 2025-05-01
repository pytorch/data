# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import Cycler, Prefetcher
from torchdata.nodes.adapters import IterableWrapper

from .utils import MockSource, run_test_save_load_state, StatefulRangeNode


class TestCycler(TestCase):
    def test_cycler_basic(self) -> None:
        # Test with a simple range
        source = IterableWrapper(range(5))
        node = Cycler(source)

        # Collect 12 items (more than in the source)
        results = []
        for _ in range(12):
            results.append(next(node))

        # First 5 should match source, then it should cycle
        expected = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
        self.assertEqual(results, expected)

        # Verify cycles counter
        self.assertEqual(node._num_cycles, 2)  # Completed 2 full cycles (5 + 5 items)

    def test_cycler_with_mock_source(self) -> None:
        num_samples = 3
        source = MockSource(num_samples=num_samples)
        node = Cycler(source)

        # Collect 8 items (more than in the source)
        results = []
        for _ in range(8):
            results.append(next(node))

        # Verify cycles counter
        self.assertEqual(node._num_cycles, 2)  # Completed 2 full cycles (3 + 3 items)

        # Check that cycling works with the mock source's data structure
        for i, result in enumerate(results):
            expected_step = i % num_samples  # Cycles every num_samples
            self.assertEqual(result["step"], expected_step)
            self.assertEqual(result["test_tensor"].item(), expected_step)
            self.assertEqual(result["test_str"], f"str_{expected_step}")

    def test_cycler_empty_source(self) -> None:
        source = IterableWrapper([])
        node = Cycler(source)

        # Trying to iterate should raise StopIteration immediately
        with self.assertRaises(StopIteration):
            next(node)

        # No cycles should have been completed for an empty source
        self.assertEqual(node._num_cycles, 0)

    def test_cycler_reset_state(self) -> None:
        source = IterableWrapper(range(3))
        node = Cycler(source)

        # Go through one full cycle and into the next
        for _ in range(4):  # 3 items in source + 1 more
            next(node)

        # Check cycles counter after one cycle
        self.assertEqual(node._num_cycles, 1)

        # Get state and reset
        state = node.state_dict()
        node.reset(state)

        # Cycles counter should be preserved after reset with state
        self.assertEqual(node._num_cycles, 1)

        # Should continue from where we left off (1st item of 2nd cycle)
        self.assertEqual(next(node), 1)
        self.assertEqual(next(node), 2)

        # Complete the second cycle and start a third
        self.assertEqual(next(node), 0)

        # Cycles counter should be updated
        self.assertEqual(node._num_cycles, 2)

    def test_counter_reset(self) -> None:
        # Test that counter is properly reset
        source = IterableWrapper(range(3))
        node = Cycler(source)

        # Go through multiple cycles
        for _ in range(7):  # 2 complete cycles + 1 item
            next(node)

        # Verify cycles counter
        self.assertEqual(node._num_cycles, 2)

        # Reset without state
        node.reset()

        # Cycles counter should be reset to 0
        self.assertEqual(node._num_cycles, 0)

        # Go through one cycle
        for _ in range(3):
            next(node)

        # Verify cycles counter after one cycle
        self.assertEqual(node._num_cycles, 0)

        next(node)
        self.assertEqual(node._num_cycles, 1)

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state(self, midpoint: int) -> None:
        # Use StatefulRangeNode like in test_header.py
        n = 50
        source = StatefulRangeNode(n=n)
        node = Cycler(source, max_cycles=2)
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state_with_prefetcher(self, midpoint: int) -> None:
        # Test save/load state with Prefetcher after Cycler
        n = 50
        source = StatefulRangeNode(n=n)
        cycler = Cycler(source, max_cycles=1)
        node = Prefetcher(cycler, prefetch_factor=2)
        run_test_save_load_state(self, node, midpoint)

    def test_cycler_with_prefetcher(self) -> None:
        # Test with Prefetcher after Cycler
        n = 5
        source = IterableWrapper(range(n))
        cycler = Cycler(source)
        node = Prefetcher(cycler, prefetch_factor=3)

        # Collect 12 items (more than in the source)
        results = []
        for _ in range(12):
            results.append(next(node))

        # First 5 should match source, then it should cycle
        expected = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
        self.assertEqual(results, expected)

        # Verify cycles counter in the cycler
        self.assertEqual(cycler._num_cycles, 2)  # Completed 2 full cycles (5 + 5 items)

    def test_cycler_with_max_cycles(self) -> None:
        # Test with max_cycles parameter
        source = IterableWrapper(range(3))
        node = Cycler(source, max_cycles=2)  # Limit to 2 cycles

        # Collect all items (should be 6 with max_cycles=2)
        results = list(node)

        # Should have exactly 2 cycles of the source
        expected = [0, 1, 2, 0, 1, 2]
        self.assertEqual(results, expected)

        # Verify cycles counter
        self.assertEqual(node._num_cycles, 2)

        # Trying to iterate more should raise StopIteration
        with self.assertRaises(StopIteration):
            next(node)

    def test_max_cycles_state_preservation(self) -> None:
        # Test that max_cycles is properly preserved in state
        source = IterableWrapper(range(3))
        node = Cycler(source, max_cycles=2)

        # Go through one full cycle and start the next
        for _ in range(3):
            next(node)

        # At this point, we've consumed [0,1,2] but haven't yet triggered the cycle
        next(node)  # This will trigger StopIteration internally and return the first item of the next cycle

        # Now we should have completed 1 cycle
        self.assertEqual(node._num_cycles, 1)

        # Get state and create a new node
        state = node.state_dict()
        new_node = Cycler(IterableWrapper(range(3)), max_cycles=5)  # Different max_cycles
        new_node.reset(state)

        # max_cycles should be preserved from state
        self.assertEqual(new_node.max_cycles, 2)

        # Should only be able to get 2 more items (remaining from 2nd cycle)
        results = []
        for _ in range(2):
            results.append(next(new_node))

        self.assertEqual(results, [1, 2])  # We already got 0 before saving state

        # Next item should be from the start of cycle 3, but we hit max_cycles=2
        with self.assertRaises(StopIteration):
            next(new_node)

        # We should have completed 2 cycles
        self.assertEqual(new_node._num_cycles, 2)
