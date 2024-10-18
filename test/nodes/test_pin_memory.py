import testslide
import unittest
import torch

from torch.testing._internal.common_utils import TEST_CUDA

from torchdata.nodes.batch import Batcher
from torchdata.nodes.map import Mapper
from torchdata.nodes.pin_memory import PinMemory
from torchdata.nodes.prefetch import Prefetcher

from .utils import Collate, MockSource


@unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
class TestPinMemory(testslide.TestCase):
    def test_pin_memory(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size)
        node = Mapper(node, Collate())
        node = PinMemory(node)
        root = Prefetcher(node, prefetch_factor=2)

        # 2 epochs
        for epoch in range(2):
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
