import itertools

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.filter import Filter

from .utils import MockSource, run_test_save_load_state, StatefulRangeNode


class TestFilter(TestCase):
    def test_filter_basic(self) -> None:
        # Test with a simple range
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x % 2 == 0)  # Keep even numbers
        
        results = list(node)
        self.assertEqual(results, [0, 2, 4, 6, 8])
        
        # Test with a different predicate
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x > 5)  # Keep numbers greater than 5
        
        results = list(node)
        self.assertEqual(results, [6, 7, 8, 9])

    def test_filter_with_mock_source(self) -> None:
        num_samples = 20
        source = MockSource(num_samples=num_samples)
        node = Filter(source, lambda x: x["step"] % 3 == 0)  # Keep items where step is divisible by 3
        
        # Test multi epoch
        for _ in range(2):
            node.reset()
            results = list(node)
            expected_steps = [i for i in range(num_samples) if i % 3 == 0]
            self.assertEqual(len(results), len(expected_steps))
            
            for i, result in enumerate(results):
                expected_step = expected_steps[i]
                self.assertEqual(result["step"], expected_step)
                self.assertEqual(result["test_tensor"].item(), expected_step)
                self.assertEqual(result["test_str"], f"str_{expected_step}")

    def test_filter_empty_result(self) -> None:
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x > 100)  # No items will pass this filter
        
        results = list(node)
        self.assertEqual(results, [])

    @parameterized.expand(itertools.product([0, 3, 7]))
    def test_save_load_state(self, midpoint: int):
        n = 50
        source = StatefulRangeNode(n=n)
        node = Filter(source, lambda x: x % 3 == 0)  # Keep items divisible by 3
        run_test_save_load_state(self, node, midpoint)

    def test_filter_reset_state(self) -> None:
        source = IterableWrapper(range(10))
        node = Filter(source, lambda x: x % 2 == 0)
        
        # Consume first two items
        self.assertEqual(next(node), 0)
        self.assertEqual(next(node), 2)
        
        # Get state and reset
        state = node.state_dict()
        node.reset(state)
        
        # Should continue from where we left off
        self.assertEqual(next(node), 4)
        self.assertEqual(next(node), 6)
        self.assertEqual(next(node), 8)
        
        # Should raise StopIteration after all items are consumed
        with self.assertRaises(StopIteration):
            next(node)
