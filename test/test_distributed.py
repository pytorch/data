# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest

from functools import partial
from unittest import TestCase

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import instantiate_parametrized_tests, IS_WINDOWS, parametrize
from torch.utils.data import DataLoader

from torchdata.dataloader2 import DataLoader2, DistributedReadingService
from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.iter.util.prefetch import PrefetchTimeoutError

TEST_MASTER_ADDR = "127.0.0.1"
DEFAULT_WORLD_SIZE = 2


if not dist.is_available():
    import sys

    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


_backends = ["gloo"]
if dist.is_mpi_available():
    _backends.append("mpi")
if dist.is_nccl_available() and torch.cuda.device_count() > 0:
    _backends.append("nccl")

backend_parametrize = parametrize("backend", _backends)


def abs_path(path):
    return os.path.join(os.path.dirname(__file__), os.path.normpath(path))


def _get_open_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


def launch_distributed_training(backend, world_size, fn):
    os.environ["MASTER_ADDR"] = TEST_MASTER_ADDR
    os.environ["MASTER_PORT"] = _get_open_port()
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

    @backend_parametrize
    def test_fullsync(self, backend) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend != "nccl" else torch.cuda.device_count()
        launch_distributed_training(backend, world_size, DistributedTest._test_fullsync)

    @staticmethod
    def _get_dataloader(data_length: int, dl2: bool, shuffle: bool, rs=None):
        data_source = IterableWrapper(list(range(data_length)))

        dp = data_source.sharding_filter()
        if shuffle:
            dp = dp.shuffle()

        if dl2:
            if rs is None:
                rs = DistributedReadingService()
            dl = DataLoader2(dp, reading_service=rs)
        else:
            dp = dp.fullsync()
            dl = DataLoader(dp)

        return dl

    @staticmethod
    def _test_distributed_training(dl2, rank, world_size, backend):
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # Use a prime number to make sure uneven data sharding
        data_length = 23

        # No shuffle
        dl = DistributedTest._get_dataloader(data_length, dl2=dl2, shuffle=False)
        res = []
        for d in dl:
            res.append(d)
            # Simulate training synchronization
            dist.barrier()
        assert sorted(res) == list(range(rank, data_length // world_size * world_size, world_size))

        # Shuffle
        dl = DistributedTest._get_dataloader(data_length, dl2=dl2, shuffle=True)
        results = []
        for _ in range(2):
            res = []
            torch.manual_seed(123)
            for d in dl:
                res.append(d)
                # Simulate training synchronization
                dist.barrier()
            results.append(res)
        assert results[0] == results[1]

        # Different seed
        res = []
        torch.manual_seed(321)
        for d in dl:
            res.append(d)
            # Simulate training synchronization
            dist.barrier()
        results.append(res)
        assert len(results[0]) == len(results[2])
        assert results[0] != results[2]

    @backend_parametrize
    def test_distributed_dl2(self, backend) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend != "nccl" else torch.cuda.device_count()
        launch_distributed_training(backend, world_size, partial(DistributedTest._test_distributed_training, True))

    @unittest.skipIf(
        IS_WINDOWS,
        "Torch Elastic is not working properly on Windows. See: https://github.com/pytorch/pytorch/issues/85427",
    )
    @backend_parametrize
    def test_elastic_training_dl2(self, backend) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend != "nccl" else torch.cuda.device_count()
        nnodes = 1
        from torch.distributed import run

        run.main(
            [
                "--run_path",
                f"--nnodes={nnodes}",
                f"--nproc_per_node={world_size}",
                abs_path("bin/elastic_training.py"),
                "--" + backend,
                "--dl2",
            ],
        )

    @backend_parametrize
    def test_distributed_dl1(self, backend) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend != "nccl" else torch.cuda.device_count()
        launch_distributed_training(backend, world_size, partial(DistributedTest._test_distributed_training, False))

    @unittest.skipIf(
        IS_WINDOWS,
        "Torch Elastic is not working properly on Windows. See: https://github.com/pytorch/pytorch/issues/85427",
    )
    @backend_parametrize
    def test_elastic_training_dl1(self, backend) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend != "nccl" else torch.cuda.device_count()
        nnodes = 1
        from torch.distributed import run

        run.main(
            [
                "--run_path",
                f"--nnodes={nnodes}",
                f"--nproc_per_node={world_size}",
                abs_path("bin/elastic_training.py"),
                "--" + backend,
                "--dl1",
            ],
        )


instantiate_parametrized_tests(DistributedTest)


if __name__ == "__main__":
    unittest.main()
