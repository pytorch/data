# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from torchdata.dataloader2.graph import DataPipe
from torchdata.dataloader2.utils import generate_random_int
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe

try:
    import numpy

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False


@dataclass(frozen=True)
class DistInfo:
    r"""
    Message class for keeping track of distributed information.

    Args:
        world_size (int): Total number of distributed nodes
        rank (int): Distributed rank for the current distributed node
    """
    world_size: int = 1
    rank: int = 0


@dataclass(frozen=True)
class WorkerInfo:
    r"""
    Message class for keeping track of worker information.

    Args:
        num_workers (int): Total number of worker processes
        worker_id (int): Worker ID for the current worker process
    """
    num_workers: int
    worker_id: int


@dataclass(frozen=True)
class _ExtraInfo:
    r"""
    Message class for extra arguments.
    """
    shared_seed: int


def process_init_fn(
    datapipe: DataPipe,
    dist_info: DistInfo,
    worker_info: WorkerInfo,
    custom_init_fn: Optional[Callable[[DataPipe, DistInfo, WorkerInfo], DataPipe]] = None,
) -> DataPipe:
    r"""
    Based on the distributed and worker information, shard the ``DataPipe`` graph dynamically.
    """
    global_worker_id = worker_info.worker_id * dist_info.world_size + dist_info.rank
    total_num_workers = worker_info.num_workers * dist_info.world_size
    torch.utils.data.graph_settings.apply_sharding(datapipe, total_num_workers, global_worker_id)

    if custom_init_fn is not None:
        datapipe = custom_init_fn(datapipe, dist_info, worker_info)
        assert isinstance(datapipe, (IterDataPipe, MapDataPipe))

    return datapipe


def process_reset_fn(
    datapipe: DataPipe,
    dist_info: DistInfo,
    worker_info: WorkerInfo,
    extra_info: _ExtraInfo,
    custom_reset_fn: Optional[Callable[[DataPipe, DistInfo, WorkerInfo], DataPipe]] = None,
) -> DataPipe:
    r"""
    Based on the distributed shared random seed and worker id, this function is used to
    reset the random state of the ``DataPipe`` graph and the global random states for ``torch``,
    ``random`` and ``numpy``.
    """
    # This function will receive worker local copy of datapipe and reset function from ``initialize_iteration``
    worker_seed_generator = torch.Generator()
    worker_seed_generator.manual_seed(extra_info.shared_seed)
    torch.utils.data.graph_settings.apply_random_seed(
        datapipe,
        worker_seed_generator,
    )
    # Set different seeds across distributed workers
    global_worker_id = worker_info.worker_id * dist_info.world_size + dist_info.rank
    worker_seed_generator.manual_seed(extra_info.shared_seed + global_worker_id)

    py_seed = generate_random_int(worker_seed_generator)
    random.seed(py_seed)

    torch_seed = generate_random_int(worker_seed_generator)
    torch.manual_seed(torch_seed)

    if HAS_NUMPY:
        # Numpy only accepts uint32 as the seed
        np_seed = generate_random_int(worker_seed_generator, torch.int32)
        if np_seed < 0:
            np_seed = 2 ** 32 + np_seed
        numpy.random.seed(np_seed)

    if custom_reset_fn is not None:
        datapipe = custom_reset_fn(datapipe, dist_info, worker_info)
        assert isinstance(datapipe, (IterDataPipe, MapDataPipe))

    return datapipe
