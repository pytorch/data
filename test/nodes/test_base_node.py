# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper

from .utils import run_test_save_load_state


class TestBaseNode(TestCase):
    def test_save_load_state(self):
        run_test_save_load_state(self, IterableWrapper(range(10)), 5)
