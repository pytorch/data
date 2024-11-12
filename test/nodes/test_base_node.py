# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.base_node import _BaseNodeIterator

from .utils import run_test_save_load_state


class TestBaseNode(TestCase):
    def test_started_finished(self) -> None:
        x = IterableWrapper(range(10))
        for _ in range(3):  # test multi-epoch
            it = iter(x)
            self.assertIsInstance(it, _BaseNodeIterator)
            self.assertFalse(it.started())
            self.assertFalse(it.finished())

            for _ in it:
                self.assertTrue(it.started())
                self.assertFalse(it.finished())

            self.assertTrue(it.started())
            self.assertTrue(it.finished())

    def test_save_load_state(self):
        run_test_save_load_state(self, IterableWrapper(range(10)), 5)
