# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.loader import Loader

from .utils import DummyIterableDataset


class TestLoader(TestCase):
    def test_loader_equal_state_dict_on_save_load(self):
        count = 10
        x = Loader(IterableWrapper(DummyIterableDataset(count, name="test")))
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

        self.assertEqual(x.state_dict(), state_dict_0)

        for i in range(count // 2):
            results_copy.append(next(it))

        self.assertEqual(len(results), count)
        self.assertEqual(len(results_copy), count)
        self.assertEqual(results[count // 2 :], results_copy[count // 2 :])
