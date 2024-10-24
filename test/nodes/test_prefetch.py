# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import testslide
import torch
from torchdata.nodes.batch import Batcher
from torchdata.nodes.prefetch import Prefetcher

from .utils import IterInitError, MockSource


class TestPrefetcher(testslide.TestCase):
    def test_prefetcher(self) -> None:
        batch_size = 6
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size, drop_last=True)
        root = Prefetcher(node, prefetch_factor=2)

        # Test multi epoch shutdown and restart
        for _ in range(2):
            results = list(root)
            self.assertEqual(len(results), 3)
            for i in range(3):
                for j in range(batch_size):
                    self.assertEqual(results[i][j]["step"], i * batch_size + j)
                    self.assertEqual(results[i][j]["test_tensor"], torch.tensor([i * batch_size + j]))
                    self.assertEqual(results[i][j]["test_str"], f"str_{i * batch_size + j}")

    def test_iter_init_error(self):
        node = IterInitError()
        root = Prefetcher(node, prefetch_factor=2)

        with self.assertRaisesRegex(ValueError, "Iter Init Error"):
            list(root)
