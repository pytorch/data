# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import testslide
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.base_node import BaseNodeIterator


class TestBaseNode(testslide.TestCase):
    def test_started_finished(self) -> None:
        x = IterableWrapper(range(10))
        for _ in range(3):  # test multi-epoch
            it: BaseNodeIterator = iter(x)
            self.assertIsInstance(it, BaseNodeIterator)
            self.assertFalse(it.started())
            self.assertFalse(it.finished())

            for _ in it:
                self.assertTrue(it.started())
                self.assertFalse(it.finished())

            self.assertTrue(it.started())
            self.assertTrue(it.finished())
