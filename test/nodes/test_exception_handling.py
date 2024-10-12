import testslide
from torchdata.nodes.batch import Batcher
from torchdata.nodes.map import ParallelMapper
from torchdata.nodes.pin_memory import PinMemory
from torchdata.nodes.prefetch import Prefetcher
from torchdata.nodes.root import Root

from .utils import MockSource, udf_raises


class TestExceptionHandling(testslide.TestCase):
    def test_exception_handling_mapper(self):
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size)
        node = ParallelMapper(node, udf_raises, num_workers=2, method="thread")
        node = PinMemory(node)
        node = Prefetcher(node, prefetch_factor=2)
        root = Root(node)

        with self.assertRaisesRegex(ValueError, "test exception"):
            print(list(root))

    def test_exception_handling_multiprocess(self):
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size)
        node = ParallelMapper(node, udf_raises, num_workers=2, method="process")
        node = PinMemory(node)
        node = Prefetcher(node, prefetch_factor=2)
        root = Root(node)

        with self.assertRaisesRegex(ValueError, "test exception"):
            print(list(root))
