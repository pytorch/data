# Copyright (c) Facebook, Inc. and its affiliates.
import os
import sys
import tempfile
import unittest

import torch.multiprocessing as mp

from torch.testing._internal.common_utils import slowTest
from torch.utils.data import DataLoader

current = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(current)
sys.path.insert(0, ROOT)

from examples.audio.librispeech import LibriSpeech


class TestAudioExamples(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _test_helper(self, fn, *args, **kwargs):
        dp = fn(*args, **kwargs)
        _ = list(dp)

    @staticmethod
    def _collate_fn(batch):
        return batch

    def _test_DL_helper(self, fn, *args, **kwargs):
        dp = fn(*args, **kwargs)
        mp.set_sharing_strategy("file_system")
        dl = DataLoader(
            dp,
            batch_size=8,
            num_workers=4,
            collate_fn=TestAudioExamples._collate_fn,
            multiprocessing_context="spawn",
        )
        for _ in dl:
            pass

    @slowTest
    def test_LibriSpeech(self) -> None:
        root = self.temp_dir.name
        for ds in ["dev-other", "train-clean-100"]:
            self._test_helper(LibriSpeech, root, ds)
            # With cache and DataLoader
            self._test_DL_helper(LibriSpeech, root, ds)


if __name__ == "__main__":
    unittest.main()
