# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest

from unittest import TestCase

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.iter.util.prefetch import PrefetchTimeoutError

TEST_MASTER_ADDR = "127.0.0.1"
TEST_MASTER_PORT = "29500"
DEFAULT_WORLD_SIZE = 2


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def launch_distributed_training(backend, world_size, fn):
    os.environ["MASTER_ADDR"] = TEST_MASTER_ADDR
    os.environ["MASTER_PORT"] = TEST_MASTER_PORT
    mp.spawn(
        fn,
        args=(
            world_size,
            backend,
        ),
        nprocs=world_size,
        join=True,
    )


class DistributedTest(TestCase):
    @staticmethod
    def _test_fullsync(rank, world_size, backend):
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # Use a prime number to make sure uneven data sharding
        data_length = 23
        dp = IterableWrapper(list(range(data_length))).sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(dp, world_size, rank)

        dp1 = dp.fullsync()
        for _ in range(2):
            res = []
            for d in dp1:
                res.append(d)
                # Simulate training synchronization
                dist.barrier()
            assert res == list(range(rank, data_length // world_size * world_size, world_size))

        # Timeout Test
        dp2 = dp.fullsync(timeout=0.01)
        try:
            for _ in range(2):
                _ = list(dp2)
        except Exception as e:
            assert isinstance(e, PrefetchTimeoutError)

    @parametrize(
        "backend",
        ["gloo", "nccl"]
        if torch.cuda.nccl.is_available([])
        else [
            "gloo",
        ],
    )
    def test_fullsync(self, backend) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend == "gloo" else torch.cuda.device_count()
        launch_distributed_training(backend, world_size, DistributedTest._test_fullsync)


instantiate_parametrized_tests(DistributedTest)


if __name__ == "__main__":
    unittest.main()
