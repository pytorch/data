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
from torchdata.dataloader2 import DataLoader2, InProcessReadingService, MultiProcessingReadingService
from torchdata.dataloader2.graph.settings import set_graph_random_seed
from torchdata.dataloader2.random import SeedGenerator
from torchdata.datapipes.iter import IterableWrapper


def _random_fn(data):
    r"""
    Used to validate the randomness of subprocess-local RNGs are set deterministically.
    """
    py_random_num = random.randint(0, 2 ** 32)
    np_random_num = np.random.randint(0, 2 ** 32, dtype=np.uint32)
    torch_random_num = torch.randint(0, 2 ** 32, size=[]).item()
    return (data, py_random_num, np_random_num, torch_random_num)


class DeterminismTest(TestCase):
    @unittest.skipIf(IS_WINDOWS, "Remove when https://github.com/pytorch/data/issues/857 is fixed")
    @parametrize("num_workers", [1, 8])
    def test_mprs_determinism(self, num_workers):
        data_length = 64
        exp = list(range(data_length))

        data_source = IterableWrapper(exp)
        dp = data_source.shuffle().sharding_filter().map(_random_fn)
        rs = MultiProcessingReadingService(num_workers=num_workers)
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

    def test_graph_random_settings(self):
        def _get_dp_seeds_after_setting(worker_id, seed=123):
            data_source = IterableWrapper(list(range(100)))
            dp0 = data_source.shuffle()
            dp1, dp2, dp3 = dp0.fork(3)
            dp1 = dp1.sharding_filter()
            dp2 = dp2.shuffle()
            dp3 = dp3.shuffle()
            dp3_ = dp3.sharding_filter()
            dp4 = dp1.zip(dp2, dp3_).shuffle()

            sg = SeedGenerator(seed).spawn(worker_id)
            set_graph_random_seed(dp4, sg)

            # same seeds, different seeds
            return (dp0._seed, dp3._seed), (dp2._seed, dp4._seed)

        ss_0_123, ds_0_123 = _get_dp_seeds_after_setting(worker_id=0, seed=123)
        ss_1_123, ds_1_123 = _get_dp_seeds_after_setting(worker_id=1, seed=123)
        self.assertEqual(ss_0_123, ss_1_123)
        self.assertNotEqual(ds_0_123, ds_1_123)

        ss_0_123_, ds_0_123_ = _get_dp_seeds_after_setting(worker_id=0, seed=123)
        self.assertEqual(ss_0_123, ss_0_123_)
        self.assertEqual(ds_0_123, ds_0_123_)

        ss_0_321, ds_0_321 = _get_dp_seeds_after_setting(worker_id=0, seed=321)
        self.assertNotEqual(ss_0_123, ss_0_321)
        self.assertNotEqual(ds_0_123, ds_0_321)

    def test_sprs_determinism(self):
        data_length = 64
        exp = list(range(data_length))

        data_source = IterableWrapper(exp)
        dp = data_source.shuffle().sharding_filter().map(_random_fn)
        rs = InProcessReadingService()
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
