# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import unittest

from unittest import TestCase

import numpy as np

import torch

from torch.testing._internal.common_utils import instantiate_parametrized_tests, IS_WINDOWS, parametrize
from torchdata.dataloader2 import DataLoader2, DistributedReadingService, PrototypeMultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper


def _random_fn(data):
    r"""
    Used to validate the randomness of subprocess-local RNGs are set deterministically.
    """
    py_random_num = random.randint(0, 2 ** 32)
    np_random_num = np.random.randint(0, 2 ** 32)
    torch_random_num = torch.randint(0, 2 ** 32, size=[]).item()
    return (data, py_random_num, np_random_num, torch_random_num)


class DeterminismTest(TestCase):
    @parametrize("num_workers", [0, 8])
    def test_proto_rs_determinism(self, num_workers):
        data_length = 64
        exp = list(range(data_length))

        data_source = IterableWrapper(exp)
        dp = data_source.shuffle().sharding_filter().map(_random_fn)
        rs = PrototypeMultiProcessingReadingService(num_workers=num_workers)
        dl = DataLoader2(dp, reading_service=rs)

        # No seed
        res = []
        for d, *_ in dl:
            res.append(d)
        self.assertEqual(sorted(res), exp)

        # Shuffle with seed
        results = []
        for _ in range(2):
            res = []
            ran_res = []
            torch.manual_seed(123)
            random.seed(123)
            np.random.seed(123)
            for d, *ran_nums in dl:
                res.append(d)
                ran_res.append(ran_nums)
            self.assertEqual(sorted(res), exp)
            results.append((res, ran_res))
        # Same seed generate the same order of data and the same random state
        self.assertEqual(results[0], results[1])

        # Different seed
        res = []
        ran_res = []
        torch.manual_seed(321)
        random.seed(321)
        np.random.seed(321)
        for d, *ran_nums in dl:
            res.append(d)
            ran_res.append(ran_nums)
        self.assertEqual(sorted(res), exp)
        # Different shuffle order
        self.assertNotEqual(results[0][0], res)
        # Different subprocess-local random state
        self.assertNotEqual(results[0][1], ran_res)


instantiate_parametrized_tests(DeterminismTest)


if __name__ == "__main__":
    unittest.main()
