# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, ReadingServiceInterface
from torchdata.dataloader2.adapter import Shuffle
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe


class AdapterTest(TestCase):
    def test_shuffle(self) -> None:
        size = 500
        dp = IterableWrapper(range(size))

        dl = DataLoader2(datapipe=dp)
        self.assertEqual(list(range(size)), list(dl))

        with self.assertWarns(Warning, msg="`shuffle=True` was set, but the datapipe does not contain a `Shuffler`."):
            dl = DataLoader2(datapipe=dp, datapipe_adapter_fn=Shuffle(True))
        self.assertNotEqual(list(range(size)), list(dl))

        dp = IterableWrapper(range(size)).shuffle()

        dl = DataLoader2(datapipe=dp)
        self.assertNotEqual(list(range(size)), list(dl))

        dl = DataLoader2(dp, Shuffle(True))
        self.assertNotEqual(list(range(size)), list(dl))

        dl = DataLoader2(dp, [Shuffle(None)])
        self.assertNotEqual(list(range(size)), list(dl))

        dl = DataLoader2(dp, [Shuffle(False)])
        self.assertEqual(list(range(size)), list(dl))
