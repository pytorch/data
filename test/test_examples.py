# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(current)
sys.path.append(ROOT)

from examples.vision.caltech101 import Caltech101
from examples.vision.caltech256 import Caltech256

try:
    import scipy  # type: ignore[import] # noqa: F401 F403

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
skipIfNoSciPy = unittest.skipIf(not HAS_SCIPY, "no scipy")


class TestVisionExamples(unittest.TestCase):
    @skipIfNoSciPy
    def test_Caltech101(self):
        path = os.path.join(ROOT, "examples", "vision", "fakedata", "caltech101")
        samples = list(Caltech101(path))
        self.assertEqual(6, len(samples))

    def test_Caltech256(self):
        path = os.path.join(ROOT, "examples", "vision", "fakedata", "caltech256")
        samples = list(Caltech256(path))
        self.assertEqual(6, len(samples))


if __name__ == "__main__":
    unittest.main()
