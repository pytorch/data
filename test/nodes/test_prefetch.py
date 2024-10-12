import testslide
import torch
from torchdata.nodes.batch import Batcher
from torchdata.nodes.prefetch import Prefetcher
from torchdata.nodes.root import Root

from .utils import MockSource


class TestPrefetcher(testslide.TestCase):
    def test_prefetcher(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size, drop_last=True)
        node = Prefetcher(node, prefetch_factor=2)
        root = Root(node)

        # Test multi epoch shutdown and restart
        for _ in range(2):
            results = list(root)
            self.assertEqual(len(results), 3)
            for i in range(3):
                for j in range(batch_size):
                    self.assertEqual(results[i][j]["step"], i * batch_size + j)
                    self.assertEqual(results[i][j]["test_tensor"], torch.tensor([i * batch_size + j]))
                    self.assertEqual(results[i][j]["test_str"], f"str_{i * batch_size + j}")
