# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.base_node import BaseNode
from torchdata.nodes.loader import Loader

from .utils import DummyIterableDataset, StatefulRange


def test_loader_correct_state_dict_at_midpoint(test, node: BaseNode, count: int):
    x = Loader(node)
    results = list(x)

    # Create an iterator at end of iteration
    it = iter(x)

    results_copy = []
    for _ in range(count // 2):
        results_copy.append(next(it))
    state_dict_0 = x.state_dict()

    x.load_state_dict(state_dict_0)

    # Create an iterator in the middle of iteration
    it = iter(x)

    test.assertEqual(x.state_dict(), state_dict_0)

    for i in range(count // 2):
        results_copy.append(next(it))

    test.assertEqual(len(results), count)
    test.assertEqual(len(results_copy), count)
    test.assertEqual(results[count // 2 :], results_copy[count // 2 :])


class TestLoader(TestCase):
    def test_loader_equal_state_dict_on_save_load(self) -> None:
        count = 10
        node = IterableWrapper(DummyIterableDataset(count))
        test_loader_correct_state_dict_at_midpoint(self, node, count)

    def test_loader_equal_state_dict_on_save_load_stateful(self) -> None:
        count = 10
        node = IterableWrapper(StatefulRange(count))
        test_loader_correct_state_dict_at_midpoint(self, node, count)
