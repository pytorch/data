# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from unittest import TestCase

import torch

from torch.testing._internal.common_utils import instantiate_parametrized_tests, IS_WINDOWS, parametrize
from torchdata.dataloader2 import DataLoader2, DistributedReadingService, PrototypeMultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper


class DeterminismTest(TestCase):
    @parametrize("num_workers", [0, 8])
    def test_proto_rs_determinism(self, num_workers):
        data_length = 64
        exp = list(range(data_length))

        data_source = IterableWrapper(exp)
        dp = data_source.shuffle().sharding_filter()
        rs = PrototypeMultiProcessingReadingService(num_workers=num_workers)
        dl = DataLoader2(dp, reading_service=rs)

        # No seed
        res = []
        for d in dl:
            res.append(d)
        self.assertEqual(sorted(res), exp)

        # Shuffle with seed
        results = []
        for _ in range(2):
            res = []
            torch.manual_seed(123)
            for d in dl:
                res.append(d)
            self.assertEqual(sorted(res), exp)
            results.append(res)
        self.assertEqual(results[0], results[1])

        # Different seed
        res = []
        torch.manual_seed(321)
        for d in dl:
            res.append(d)
        self.assertEqual(sorted(res), exp)
        self.assertNotEqual(results[0], res)


instantiate_parametrized_tests(DeterminismTest)


if __name__ == "__main__":
    unittest.main()
