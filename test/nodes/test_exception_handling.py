import unittest

import testslide
from torch.testing._internal.common_utils import TEST_CUDA
from torchdata.nodes.batch import Batcher
from torchdata.nodes.map import Mapper, ParallelMapper
from torchdata.nodes.pin_memory import PinMemory
from torchdata.nodes.prefetch import Prefetcher

from .utils import MockSource, udf_raises


class TestExceptionHandling(testslide.TestCase):
    def _test_exception_handling_mapper(self, pin_memory, method):
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size)
        node = ParallelMapper(node, udf_raises, num_workers=2, method=method)
        node = Mapper(node, udf_raises)
        if pin_memory:
            node = PinMemory(node)
        node = Prefetcher(node, prefetch_factor=2)

        with self.assertRaisesRegex(ValueError, "test exception"):
            print(list(node))

    def test_exception_handling_mapper(self):
        self._test_exception_handling_mapper(False, "thread")

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_exception_handling_mapper_cuda(self):
        self._test_exception_handling_mapper(True, "thread")

    def test_exception_handling_mapper_multiprocess(self):
        self._test_exception_handling_mapper(False, "process")

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_exception_handling_mapper_multiprocess_cuda(self):
        self._test_exception_handling_mapper(True, "process")
