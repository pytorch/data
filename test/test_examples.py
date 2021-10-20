# Copyright (c) Facebook, Inc. and its affiliates.
import os
import sys
import unittest

from torch.testing._internal.common_utils import slowTest

current = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(current)
sys.path.append(ROOT)

from examples.text.ag_news import AG_NEWS
from examples.text.amazonreviewpolarity import AmazonReviewPolarity
from examples.text.imdb import IMDB
from examples.text.squad1 import SQuAD1
from examples.text.squad2 import SQuAD2
from examples.text.sst2 import SST2
from examples.vision.caltech101 import Caltech101
from examples.vision.caltech256 import Caltech256

try:
    import scipy  # type: ignore[import] # noqa: F401 F403

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
skipIfNoSciPy = unittest.skipIf(not HAS_SCIPY, "no scipy")

try:
    import PIL  # type: ignore[import] # noqa: F401 F403

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
skipIfNoPillow = unittest.skipIf(not HAS_PILLOW, "no pillow")


@skipIfNoSciPy
@skipIfNoPillow
class TestVisionExamples(unittest.TestCase):
    def test_Caltech101(self) -> None:
        path = os.path.join(ROOT, "examples", "vision", "fakedata", "caltech101")
        samples = list(Caltech101(path))
        self.assertEqual(6, len(samples))

    def test_Caltech256(self) -> None:
        path = os.path.join(ROOT, "examples", "vision", "fakedata", "caltech256")
        samples = list(Caltech256(path))
        self.assertEqual(6, len(samples))


# TODO: Replace the following tests with the corresponding tests in TorchText
class TestTextExamples(unittest.TestCase):
    def _test_helper(self, fn):
        dp = fn()
        for stage_dp in dp:
            _ = list(stage_dp)

    @slowTest
    def test_AG_NEWS(self) -> None:
        self._test_helper(AG_NEWS)

    @slowTest
    def test_AmazonReviewPolarity(self) -> None:
        self._test_helper(AmazonReviewPolarity)

    @slowTest
    def test_IMDB(self) -> None:
        self._test_helper(IMDB)

    @slowTest
    def test_SQuAD1(self) -> None:
        self._test_helper(SQuAD1)

    @slowTest
    def test_SQuAD2(self) -> None:
        self._test_helper(SQuAD2)

    @slowTest
    def test_SST2(self) -> None:
        self._test_helper(SST2)


if __name__ == "__main__":
    unittest.main()
