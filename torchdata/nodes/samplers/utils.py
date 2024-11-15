# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed as dist


class StopCriteria:
    """
    Stopping criteria for the dataset samplers.

    1) CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED: Stop once the last unseen dataset is exhausted.
        All datasets are seen at least once. In certain cases, some datasets may be
        seen more than once when there are still non-exhausted datasets.

    2) ALL_DATASETS_EXHAUSTED: Stop once all have the datasets are exhausted. Each
        dataset is seen exactly once. No wraparound or restart will be performed.

    3) FIRST_DATASET_EXHAUSTED: Stop when the first dataset is exhausted.
    """

    CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED = "CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED"
    ALL_DATASETS_EXHAUSTED = "ALL_DATASETS_EXHAUSTED"
    FIRST_DATASET_EXHAUSTED = "FIRST_DATASET_EXHAUSTED"


def _get_rank_seed(seed: int, generator_rank: torch.Generator, rank: int, world_size: int) -> int:
    generator_rank.manual_seed(seed * world_size + rank)
    return int(torch.randint(0, 2 ** 32 - 1, size=(1,), generator=generator_rank).item())


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
        rank = os.environ.get("RANK", "0")
        world_size = os.environ.get("WORLD_SIZE", "1")
        try:
            rank = int(rank)
            world_size = int(world_size)
        except ValueError:
            rank = 0
            world_size = 1

    if rank >= world_size or rank < 0:
        raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]")

    return rank, world_size
