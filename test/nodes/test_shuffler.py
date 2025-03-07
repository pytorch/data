import itertools
import random

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes import Shuffler
from torchdata.nodes.adapters import IterableWrapper

from .utils import MockSource, run_test_save_load_state, StatefulRangeNode


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

            # Verify shuffled counter
            self.assertEqual(node._num_shuffled, 10)

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

        # Verify shuffled counter
        self.assertEqual(node._num_shuffled, num_samples)

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

        # Verify shuffled counter with empty source
        self.assertEqual(node._num_shuffled, 0)

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state(self, midpoint: int) -> None:
        n = 50
        source = StatefulRangeNode(n=n)
        node = Shuffler(source, buffer_size=10, seed=42)
        run_test_save_load_state(self, node, midpoint)

    def test_shuffler_reset_state(self) -> None:
        # Use a fixed seed for deterministic testing
        random.seed(42)

        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42)

        # Consume first three items
        shuffled_items = [next(node) for _ in range(3)]

        # Check counter after consuming items
        self.assertEqual(node._num_shuffled, 3)

        # Get state and reset
        state = node.state_dict()
        node.reset(state)

        # Counter should be preserved after reset with state
        self.assertEqual(node._num_shuffled, 3)

        # Get next few items
        more_items = [next(node) for _ in range(7)]

        # All 10 items should be unique
        all_items = shuffled_items + more_items
        self.assertEqual(len(set(all_items)), 10)
        self.assertEqual(sorted(all_items), list(range(10)))

        # Counter should be updated
        self.assertEqual(node._num_shuffled, 10)

        # Should raise StopIteration after all items are consumed
        with self.assertRaises(StopIteration):
            next(node)

    def test_counter_reset(self) -> None:
        # Test that counter is properly reset
        source = IterableWrapper(range(10))
        node = Shuffler(source, buffer_size=5, seed=42)

        # Consume all items
        list(node)

        # Verify counter after first pass
        self.assertEqual(node._num_shuffled, 10)

        # Reset without state
        node.reset()

        # Counter should be reset to 0
        self.assertEqual(node._num_shuffled, 0)

        # Consume some items
        for _ in range(3):
            next(node)

        # Verify counter after partial consumption
        self.assertEqual(node._num_shuffled, 3)

    def test_invalid_input(self) -> None:
        # Test with invalid buffer size
        source = IterableWrapper(range(10))

        # Buffer size must be at least 1
        with self.assertRaises(ValueError):
            Shuffler(source, buffer_size=0)

        with self.assertRaises(ValueError):
            Shuffler(source, buffer_size=-3)
