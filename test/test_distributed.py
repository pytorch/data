# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import queue
import random
import socket
import sys
import unittest

from functools import partial
from unittest import TestCase

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import instantiate_parametrized_tests, IS_WINDOWS, parametrize
from torch.utils.data import DataLoader

from torchdata.dataloader2 import DataLoader2, DistributedReadingService, PrototypeMultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.iter.util.prefetch import PrefetchTimeoutError

TEST_MASTER_ADDR = "127.0.0.1"
DEFAULT_WORLD_SIZE = 2


if not dist.is_available():
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
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


class TerminateSignal:
    pass


# TODO(ejguan): Use queue for all distributed tests
def launch_distributed_training(backend, world_size, *args, fn):
    os.environ["MASTER_ADDR"] = TEST_MASTER_ADDR
    os.environ["MASTER_PORT"] = _get_open_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = []
    for rank in range(world_size):
        p = ctx.Process(
            target=fn,
            args=(
                rank,
                world_size,
                backend,
                q,
                *args,
            ),
        )
        p.start()
        ps.append(p)
    res = []
    while True:
        try:
            d = q.get()
            if isinstance(d, TerminateSignal):
                break
            res.append(d)
        except queue.Empty:
            continue
    for p in ps:
        p.join()
    return res


def _dist_iterate_one_epoch(dl, seed=None):
    r"""
    Iterate a full epoch of DataLoader and set seeds for global RNGs if provided.
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    res = []
    for d in dl:
        res.append(d)
        # Simulate training synchronization
        dist.barrier()
    return res


def _finalize_distributed_queue(rank, q):
    r"""
    Synchronize all distributed processes to guarantee all data have been put into
    the Multiprocessing Queue.
    """
    pg = dist.new_group(backend="gloo")
    end_tensor = torch.tensor([rank], dtype=torch.int64)
    dist.all_reduce(end_tensor, group=pg)
    if rank == 0:
        q.put(TerminateSignal())

    dist.destroy_process_group(pg)


def _random_fn(data):
    r"""
    Used to validate the randomness of subprocess-local RNGs are set deterministically.
    """
    py_random_num = random.randint(0, 2 ** 32)
    np_random_num = np.random.randint(0, 2 ** 32)
    torch_random_num = torch.randint(0, 2 ** 32, size=[]).item()
    return (data, py_random_num, np_random_num, torch_random_num)


def _test_proto_distributed_training(rank, world_size, backend, q, num_workers):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # Balanced data
    data_length = world_size * 8
    if num_workers > 0:
        data_length *= num_workers

    data_source = IterableWrapper(list(range(data_length)))
    dp = data_source.shuffle().sharding_filter().map(_random_fn)
    rs = PrototypeMultiProcessingReadingService(num_workers=num_workers)
    dl = DataLoader2(dp, reading_service=rs)

    # No seed
    res = _dist_iterate_one_epoch(dl, seed=None)
    q.put((0, rank, res))

    # Shuffle with seed
    for epoch in range(2):
        res = _dist_iterate_one_epoch(dl, seed=123)
        q.put((epoch + 1, rank, res))

    # Different seed
    res = _dist_iterate_one_epoch(dl, seed=321)
    q.put((3, rank, res))

    _finalize_distributed_queue(rank, q)
    dl.shutdown()


class DistributedTest(TestCase):
    @staticmethod
    def _test_fullsync(rank, world_size, backend, q):
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # Use a prime number to make sure uneven data sharding
        data_length = 23
        dp = IterableWrapper(list(range(data_length))).sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(dp, world_size, rank)

        dp1 = dp.fullsync()
        for _ in range(2):
            res = _dist_iterate_one_epoch(dp1)
            assert res == list(range(rank, data_length // world_size * world_size, world_size))

        # Timeout Test
        dp2 = dp.fullsync(timeout=0.01)
        try:
            for _ in range(2):
                _ = list(dp2)
        except Exception as e:
            assert isinstance(e, PrefetchTimeoutError)

        _finalize_distributed_queue(rank, q)

    @backend_parametrize
    def test_fullsync(self, backend) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend != "nccl" else torch.cuda.device_count()
        launch_distributed_training(backend, world_size, fn=DistributedTest._test_fullsync)

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
    def _test_distributed_training(dl2, rank, world_size, backend, q):
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # Use a prime number to make sure uneven data sharding
        data_length = 23

        # No shuffle
        dl = DistributedTest._get_dataloader(data_length, dl2=dl2, shuffle=False)
        res = _dist_iterate_one_epoch(dl)
        assert sorted(res) == list(range(rank, data_length // world_size * world_size, world_size))

        # Shuffle
        dl = DistributedTest._get_dataloader(data_length, dl2=dl2, shuffle=True)
        results = []
        for _ in range(2):
            res = _dist_iterate_one_epoch(dl, seed=123)
            results.append(res)
        assert results[0] == results[1]

        # Different seed
        res = _dist_iterate_one_epoch(dl, seed=321)
        results.append(res)
        assert len(results[0]) == len(results[2])
        assert results[0] != results[2]

        _finalize_distributed_queue(rank, q)
        if dl2:
            dl.shutdown()

    @backend_parametrize
    def test_distributed_dl2(self, backend) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend != "nccl" else torch.cuda.device_count()
        launch_distributed_training(backend, world_size, fn=partial(DistributedTest._test_distributed_training, True))

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
        launch_distributed_training(backend, world_size, fn=partial(DistributedTest._test_distributed_training, False))

    @unittest.skipIf(
        IS_WINDOWS,
        "Torch Elastic is not working properly on Windows. See: https://github.com/pytorch/pytorch/issues/85427",
    )
    @unittest.skipIf(sys.version_info < (3, 8), "Torch Elastic requires Python >= 3.8")
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

    @unittest.skipIf(IS_WINDOWS, "Remove when https://github.com/pytorch/data/issues/857 is fixed")
    @backend_parametrize
    @parametrize("num_workers", [0, 8])
    def test_proto_rs_dl2(self, backend, num_workers) -> None:
        world_size = DEFAULT_WORLD_SIZE if backend != "nccl" else torch.cuda.device_count()
        res = launch_distributed_training(backend, world_size, num_workers, fn=_test_proto_distributed_training)
        result = ({}, {}, {}, {})
        for epoch, rank, r in res:
            d, *ran_nums = list(zip(*r))
            result[epoch][rank] = (d, ran_nums)
        # Same seed generate the same order of data and the same random state
        self.assertEqual(result[1], result[2])
        # Different seeds
        for rank in range(world_size):
            # Different shuffle order
            self.assertNotEqual(result[1][rank][0], result[3][rank][0])
            # Different subprocess-local random state
            self.assertNotEqual(result[1][rank][1], result[3][rank][1])

        # Mutually exclusive and collectively exhaustive with/without seed
        data_length = world_size * 8
        if num_workers > 0:
            data_length *= num_workers
        exp = list(range(data_length))
        for res in result:
            concat_res = []
            for r in res.values():
                concat_res.extend(r[0])
            self.assertEqual(sorted(concat_res), exp)


instantiate_parametrized_tests(DistributedTest)


if __name__ == "__main__":
    unittest.main()
