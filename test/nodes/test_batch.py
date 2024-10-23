# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import testslide
import torch
from torchdata.nodes.batch import Batcher

from .utils import MockSource


class TestBatcher(testslide.TestCase):
    def test_batcher(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size, drop_last=True)

        results = list(node)
        self.assertEqual(len(results), 3)
        for i in range(3):
            for j in range(batch_size):
                self.assertEqual(results[i][j]["step"], i * batch_size + j)
                self.assertEqual(results[i][j]["test_tensor"], torch.tensor([i * batch_size + j]))
                self.assertEqual(results[i][j]["test_str"], f"str_{i * batch_size + j}")

    def test_batcher_drop_last_false(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        root = Batcher(src, batch_size=batch_size, drop_last=False)

        results = list(root)
        self.assertEqual(len(results), 4)
        for i in range(4):
            n = batch_size if i < 3 else 2
            for j in range(n):
                self.assertEqual(results[i][j]["step"], i * batch_size + j)
                self.assertEqual(results[i][j]["test_tensor"], torch.tensor([i * batch_size + j]))
                self.assertEqual(results[i][j]["test_str"], f"str_{i * batch_size + j}")
