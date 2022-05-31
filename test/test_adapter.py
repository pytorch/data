# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from unittest import TestCase

import torch

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, ReadingServiceInterface
from torchdata.dataloader2.adapter import Shuffle
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe


class AdapterTest(TestCase):
    def test_shuffle(self) -> None:
        size = 500
        dp = IterableWrapper(range(size))

        dl = DataLoader2(datapipe=dp)
        self.assertEqual(list(range(size)), list(dl))

        with warnings.catch_warnings(record=True) as wa:
            dl = DataLoader2(datapipe=dp, datapipe_adapter_fn=Shuffle(True))
            self.assertNotEqual(list(range(size)), list(dl))
            self.assertEqual(1, len(wa))

        dp = IterableWrapper(range(size)).shuffle()

        dl = DataLoader2(datapipe=dp)
        self.assertNotEqual(list(range(size)), list(dl))

        dl = DataLoader2(dp, Shuffle(True))
        self.assertNotEqual(list(range(size)), list(dl))

        dl = DataLoader2(dp, [Shuffle(None)])
        self.assertNotEqual(list(range(size)), list(dl))

        dl = DataLoader2(dp, [Shuffle(False)])
        self.assertEqual(list(range(size)), list(dl))

    def test_pin_memory(self):
        size = 10
        dp = IterableWrapper(range(size)).map(lambda x: {"a": torch.Tensor(x)}).pin_memory()
        self.assertEqual(list(range(size)), list(dp))
