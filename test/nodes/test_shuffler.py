import itertools
import random

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import Shuffler
from torchdata.nodes.adapters import IterableWrapper

from .utils import MockSource, StatefulRangeNode


class TestShuffler(TestCase):
    def test_shuffler_basic(self) -> None:
        # Test with a simple range and different buffer sizes
        for buffer_size in [1, 3, 10]:
            source = IterableWrapper(range(10))
            node = Shuffler(source, buffer_size=buffer_size, seed=42)

            results = list(node)

            # Results should contain all original items
            self.assertEqual(sorted(results), list(range(10)))

            # With buffer_size=1, no shuffling should occur
            if buffer_size == 1:
                self.assertEqual(results, list(range(10)))
            else:
                # With buffer_size > 1, results should be shuffled
                self.assertNotEqual(results, list(range(10)))

            # Verify yielded counter
            self.assertEqual(node._num_yielded, 10)

    def test_shuffler_deterministic(self) -> None:
        # Test that results are deterministic with the same seed
        source1 = IterableWrapper(range(20))
        source2 = IterableWrapper(range(20))

        node1 = Shuffler(source1, buffer_size=10, seed=12345)
        node2 = Shuffler(source2, buffer_size=10, seed=12345)

        results1 = list(node1)
        results2 = list(node2)

        # Results should be identical with the same seed
        self.assertEqual(results1, results2)

        # Results should be different with different seeds
        source3 = IterableWrapper(range(20))
        node3 = Shuffler(source3, buffer_size=10, seed=54321)
        results3 = list(node3)

        # Very unlikely that different seeds produce same shuffle
        self.assertNotEqual(results1, results3)

    def test_shuffler_with_mock_source(self) -> None:
        num_samples = 20
        source = MockSource(num_samples=num_samples)
        node = Shuffler(source, buffer_size=10, seed=42)

        results = list(node)
        self.assertEqual(len(results), num_samples)

        # Verify yielded counter
        self.assertEqual(node._num_yielded, num_samples)

        # Check that all items are present
        step_values = [result["step"] for result in results]
        self.assertEqual(sorted(step_values), list(range(num_samples)))

        # With seed=42, result should be shuffled
        self.assertNotEqual(step_values, list(range(num_samples)))

    def test_shuffler_empty_source(self) -> None:
        source = IterableWrapper([])
        node = Shuffler(source, buffer_size=5, seed=42)

        results = list(node)
        self.assertEqual(results, [])

        # Verify yielded counter with empty source
        self.assertEqual(node._num_yielded, 0)

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state(self, midpoint: int) -> None:
        """Test that saving and loading state preserves the deterministic sequence."""
        n = 20
        source = StatefulRangeNode(n=n)
        node = Shuffler(source, buffer_size=5, seed=42)

        # First, collect all items without interruption
        source_copy = StatefulRangeNode(n=n)
        node_copy = Shuffler(source_copy, buffer_size=5, seed=42)
        expected_items = list(node_copy)

        # Now collect with save/load at midpoint
        items_before = []
        for _ in range(midpoint):
            try:
                items_before.append(next(node))
            except StopIteration:
                break

        # Save state and create a new node
        state = node.state_dict()
        new_source = StatefulRangeNode(n=n)
        new_node = Shuffler(new_source, buffer_size=5, seed=42)
        new_node.reset(state)

        # Collect remaining items
        items_after = []
        try:
            while True:
                items_after.append(next(new_node))
        except StopIteration:
            pass

        # Combined sequence should match the expected sequence
        combined = items_before + items_after

        # The combined sequence will have fewer items than expected because
        # we don't preserve the buffer in the state. We expect to lose
        # approximately buffer_size items.
        buffer_size = 5
        self.assertLessEqual(len(expected_items) - len(combined), buffer_size)

        # For dictionary items, extract values for comparison
        if expected_items and isinstance(expected_items[0], dict):
            # Assume items have a standard format (e.g., with a 'value' or 'index' key)
            keys_to_check = expected_items[0].keys()

            # Verify each item has the same keys
            for item in combined:
                self.assertEqual(set(item.keys()), set(keys_to_check))

            # Check that combined items are a subset of expected_items
            # We'll do this by creating a string representation of each dict
            combined_str = [str(sorted(item.items())) for item in combined]
            expected_str = [str(sorted(item.items())) for item in expected_items]

            # Every item in combined should be in expected
            for item in combined_str:
                self.assertIn(item, expected_str)
        else:
            # For non-dictionary items, check subset relationship
            for item in combined:
                self.assertIn(item, expected_items)

        # Verify that the first items match exactly (deterministic order)
        if midpoint > 0:
            self.assertEqual(items_before, expected_items[:midpoint])

    def test_shuffler_reset_state(self) -> None:
        # This test verifies that after resetting with a state,
        # the counter is preserved but the buffer is empty

        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42)

        # Consume first three items
        for _ in range(3):
            next(node)

        # Check counter after consuming items
        self.assertEqual(node._num_yielded, 3)

        # Get state and reset
        state = node.state_dict()

        # Create a new node with a fresh source
        new_source = IterableWrapper(range(10))
        new_node = Shuffler(new_source, buffer_size=5, seed=42)
        new_node.reset(state)

        # Counter should be preserved after reset with state
        self.assertEqual(new_node._num_yielded, 3)

        # Since we don't preserve the buffer in the state,
        # we should be able to get the remaining items from the source
        # (the source state is preserved, so it starts from where it left off)
        items = []
        try:
            while True:
                items.append(next(new_node))
        except StopIteration:
            pass

        # We should get some remaining items
        self.assertGreater(len(items), 0)

        # The items should be a subset of the range
        for item in items:
            self.assertIn(item, range(10))

        # Counter should reflect total items yielded
        self.assertEqual(new_node._num_yielded, 3 + len(items))

    def test_counter_reset(self) -> None:
        # Test that counter is properly reset
        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42)

        # Consume all items
        list(node)

        # Verify counter after first pass
        self.assertEqual(node._num_yielded, 10)

        # Reset without state
        node.reset()

        # Counter should be reset to 0
        self.assertEqual(node._num_yielded, 0)

        # Consume some items
        for _ in range(3):
            next(node)

        # Verify counter after partial consumption
        self.assertEqual(node._num_yielded, 3)

    def test_invalid_input(self) -> None:
        # Test with invalid buffer size
        source = IterableWrapper(range(10))

        # Buffer size must be at least 1
        with self.assertRaises(ValueError):
            Shuffler(source, buffer_size=0)

        with self.assertRaises(ValueError):
            Shuffler(source, buffer_size=-3)
