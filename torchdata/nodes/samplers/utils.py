# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed as dist


def _get_rank_seed(seed: int, generator_rank: torch.Generator, rank: int, world_size: int, epoch: int) -> int:
    generator_rank.manual_seed(seed * world_size + rank)
    return int(torch.randint(0, 2 ** 32 - 1, size=(epoch + 1,), generator=generator_rank)[-1].item())


def get_rank_and_world_size() -> tuple[int, int]:
    """
    Returns the rank and world size of the current process.
    If distributed is initialized, returns the rank and world size from the distributed environment.
    If distributed is not initialized, returns the rank and world size from the environment variables.
    If neither distributed nor environment variables are set, returns a rank of 0 and a world size of 1.
    """
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        _rank = os.environ.get("RANK", "0")
        _world_size = os.environ.get("WORLD_SIZE", "1")
        try:
            rank = int(_rank)
            world_size = int(_world_size)
        except ValueError:
            rank = 0
            world_size = 1

    if rank >= world_size or rank < 0:
        raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]")

    return rank, world_size
