# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(current)
sys.path.append(ROOT)

from torchdata.datapipes.iter import IterableWrapper

class TestClass(unittest.TestCase):
    def test_wrapper(self):
        dp = IterableWrapper(range(10))
        self.assertEqual(list(range(10)), list(dp))

if __name__ == '__main__':
    unittest.main()