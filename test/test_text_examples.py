# Copyright (c) Facebook, Inc. and its affiliates.
import os
import sys
import unittest

import torch.multiprocessing as mp

from torch.testing._internal.common_utils import slowTest
from torch.utils.data import DataLoader

current = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(current)
sys.path.insert(0, ROOT)

from torchtext.datasets import AG_NEWS, AmazonReviewPolarity, IMDB, SQuAD1, SQuAD2, SST2


# TODO(124): Replace the following tests with the corresponding tests in TorchText
class TestTextExamples(unittest.TestCase):
    def _test_helper(self, fn):
        dp = fn()
        for stage_dp in dp:
            _ = list(stage_dp)

    @staticmethod
    def _collate_fn(batch):
        return batch

    def _test_DL_helper(self, fn):
        mp.set_sharing_strategy("file_system")
        dp = fn()
        for stage_dp in dp:
            dl = DataLoader(
                stage_dp,
                batch_size=8,
                num_workers=4,
                collate_fn=TestTextExamples._collate_fn,
                multiprocessing_context="spawn",
            )
            _ = list(dl)

    def test_SST(self) -> None:
        self._test_helper(SST2)
        self._test_DL_helper(SST2)

    def test_AG_NEWS(self) -> None:
        self._test_helper(AG_NEWS)
        self._test_DL_helper(AG_NEWS)

    @slowTest
    def test_AmazonReviewPolarity(self) -> None:
        self._test_helper(AmazonReviewPolarity)
        self._test_DL_helper(AmazonReviewPolarity)

    @slowTest
    def test_IMDB(self) -> None:
        self._test_helper(IMDB)
        self._test_DL_helper(IMDB)

    def test_SQuAD1(self) -> None:
        self._test_helper(SQuAD1)
        self._test_DL_helper(SQuAD1)

    def test_SQuAD2(self) -> None:
        self._test_helper(SQuAD2)
        self._test_DL_helper(SQuAD2)


if __name__ == "__main__":
    unittest.main()
