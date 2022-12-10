# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import torch
import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torchdata.dataloader2 import DataLoader2, DistributedReadingService
from torchdata.datapipes.iter import IterableWrapper


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


@record
def main(backend, dl2):
    dist.init_process_group(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Use a prime number to make sure uneven data sharding
    data_length = 23

    # No Shuffle
    dl = _get_dataloader(data_length, dl2=dl2, shuffle=False)
    res = []
    for d in dl:
        res.append(d)
        # Simulate training synchronization
        dist.barrier()
    assert sorted(res) == list(range(rank, data_length // world_size * world_size, world_size))

    # Shuffle
    dl = _get_dataloader(data_length, dl2=dl2, shuffle=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elastic Training")
    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument("--gloo", action="store_true", help="GLOO backend")
    backend_group.add_argument("--nccl", action="store_true", help="NCCL backend")
    backend_group.add_argument("--mpi", action="store_true", help="MPI backend")
    dl_group = parser.add_mutually_exclusive_group(required=True)
    dl_group.add_argument("--dl1", action="store_true", help="DataLoader")
    dl_group.add_argument("--dl2", action="store_true", help="DataLoader2")

    args = parser.parse_args()

    backend = "gloo"
    if args.nccl:
        backend = "nccl"
    elif args.mpi:
        backend = "mpi"

    dl2 = True
    if args.dl1:
        dl2 = False

    main(backend, dl2)
