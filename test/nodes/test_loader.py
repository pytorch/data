# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper

from .utils import DummyIterableDataset, StatefulRange, test_loader_correct_state_dict_at_midpoint


class TestLoader(TestCase):
    def test_loader_equal_state_dict_on_save_load(self) -> None:
        count = 10
        node = IterableWrapper(DummyIterableDataset(count))
        test_loader_correct_state_dict_at_midpoint(self, node, count)

    def test_loader_equal_state_dict_on_save_load_stateful(self) -> None:
        count = 10
        node = IterableWrapper(StatefulRange(count))
        test_loader_correct_state_dict_at_midpoint(self, node, count)
