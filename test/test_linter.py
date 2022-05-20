# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchdata.dataloader2.linter import _check_shuffle_before_sharding

from torchdata.datapipes.iter import IterableWrapper, ShardingFilter, Shuffler


def dummy_fn(x):
    return x


class LinterTest(unittest.TestCase):
    def test_sharding_shuffle(self):
        source_dp = IterableWrapper(list(range(20)))

        # Single path
        dp = source_dp.map(dummy_fn).shuffle()
        self.assertTrue(_check_shuffle_before_sharding(dp))
        dp = source_dp.map(dummy_fn)
        self.assertTrue(_check_shuffle_before_sharding(dp))

        dp = source_dp.map(dummy_fn).shuffle().sharding_filter()
        self.assertTrue(_check_shuffle_before_sharding(dp))

        dp = source_dp.map(dummy_fn).sharding_filter()
        self.assertFalse(_check_shuffle_before_sharding(dp))

        dp = source_dp.map(dummy_fn).sharding_filter().shuffle()
        self.assertFalse(_check_shuffle_before_sharding(dp))

        # Multi pathes
        def _multi_path_dp_1(shuffle):
            s_dp = source_dp.shuffle() if shuffle else source_dp
            dp1, dp2 = s_dp.unzip(2)
            dp1 = dp1.sharding_filter()
            dp2 = dp2.map(dummy_fn).sharding_filter()
            dp = dp1.zip(dp2)
            return dp

        self.assertTrue(_check_shuffle_before_sharding(_multi_path_dp_1(True)))
        self.assertFalse(_check_shuffle_before_sharding(_multi_path_dp_1(False)))

        def _multi_path_dp_2(shuffle):
            s_dp = source_dp.shuffle() if shuffle else source_dp
            dp1, dp2 = s_dp.unzip(2)
            dp1 = dp1.map(dummy_fn)
            dp = dp1.zip(dp2).sharding_filter()
            return dp

        self.assertTrue(_check_shuffle_before_sharding(_multi_path_dp_2(True)))
        self.assertFalse(_check_shuffle_before_sharding(_multi_path_dp_2(False)))

        def _multi_path_dp_3(shuffle):
            dp1, dp2 = source_dp.unzip(2)
            dp1 = dp1.shuffle() if shuffle else dp1
            dp1 = dp1.map(dummy_fn).sharding_filter()
            dp2 = dp2.shuffle() if shuffle else dp1
            dp2 = dp2.sharding_filter()
            dp = dp1.zip(dp2).map(dummy_fn)
            return dp

        self.assertTrue(_check_shuffle_before_sharding(_multi_path_dp_3(True)))
        self.assertFalse(_check_shuffle_before_sharding(_multi_path_dp_3(False)))

        # Partial paths
        dp1, dp2 = source_dp.unzip(2)
        dp1 = dp1.shuffle().map(dummy_fn)
        dp = dp1.zip(dp2).sharding_filter()

        self.assertFalse(_check_shuffle_before_sharding(dp))


if __name__ == "__main__":
    unittest.main()
