# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch

from torch.utils.data.datapipes.iter.grouping import SHARDING_PRIORITIES

from torchdata.dataloader2 import communication
from torchdata.dataloader2.graph import DataPipe, find_dps, replace_dp, traverse_dps
from torchdata.dataloader2.utils import generate_random_int
from torchdata.dataloader2.utils.non_shardable import _DummyIterDataPipe, find_shardable_branches
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe

try:
    import numpy

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False


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
class _DistInfo:
    r"""
    Message class for distribtued information.

    Args:
        shared_seed: Distributed shared random seed
        world_size (int): Total number of distributed nodes
        rank (int): Distributed rank for the current distributed node
    """
    shared_seed: int
    world_size: int = 1
    rank: int = 0


def process_init_fn(
    datapipe: DataPipe,
    worker_info: WorkerInfo,
    custom_init_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]] = None,
) -> DataPipe:
    r"""
    Based on the worker information, shard the ``DataPipe`` graph dynamically.
    """
    # Find if there is non-sharding process
    graph = traverse_dps(datapipe)
    non_shardable_dp = find_dps(graph, _DummyIterDataPipe)  # type: ignore

    # There are two cases for DataPipe graph in terms of mp sharding:
    # 1) All DataPipes are shardable, apply mp sharding to the whole graph
    if len(non_shardable_dp) == 0:
        torch.utils.data.graph_settings.apply_sharding(
            datapipe, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
        )
    # 2) There is non-shardable DataPipe. Since we have replaced the lowest common
    #    ancestor by a `_DummyIterDataPipe`, we would only apply mp sharding
    #    to the branch of DataPipes that doesn't contain `_DummyIterDataPipe`.
    else:
        assert len(non_shardable_dp) == 1
        shardable_branches = find_shardable_branches(graph)
        for dp in shardable_branches:
            torch.utils.data.graph_settings.apply_sharding(
                dp, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
            )

        req_queue = non_shardable_dp[0].req_queue
        res_queue = non_shardable_dp[0].res_queue

        queue_wrapper = communication.iter.QueueWrapper(
            communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
        )
        non_sharding_process_dp = communication.iter._IterateQueueDataPipes([queue_wrapper])
        graph = replace_dp(graph, non_shardable_dp[0], non_sharding_process_dp)
        datapipe = list(graph.values())[0][0]

    if custom_init_fn is not None:
        datapipe = custom_init_fn(datapipe, worker_info)
        assert isinstance(datapipe, (IterDataPipe, MapDataPipe))

    return datapipe


def _set_global_random_state(seed_generator: torch.Generator) -> None:
    py_seed = generate_random_int(seed_generator)
    random.seed(py_seed)

    torch_seed = generate_random_int(seed_generator)
    torch.manual_seed(torch_seed)

    if HAS_NUMPY:
        # Numpy only accepts uint32 as the seed
        np_seed = generate_random_int(seed_generator, torch.int32)
        if np_seed < 0:
            np_seed = 2 ** 32 + np_seed
        numpy.random.seed(np_seed)


def non_sharding_process_reset_fn(
    datapipe: DataPipe,
    worker_info: WorkerInfo,
    dist_info: _DistInfo,
) -> DataPipe:
    r"""
    Based on the distributed shared random seed, this function is used to set the random state
    of the ``DataPipe`` graph and the global random states for the non-sharding process.
    This function would guarantee that all distributed non-sharding process share the
    same random states to ensure the same shuffle order.
    """
    worker_seed_generator = torch.Generator()
    worker_seed_generator.manual_seed(dist_info.shared_seed)
    torch.utils.data.graph_settings.apply_random_seed(
        datapipe,
        worker_seed_generator,
    )

    # Set global random states
    _set_global_random_state(worker_seed_generator)

    return datapipe


def process_reset_fn(
    datapipe: DataPipe,
    worker_info: WorkerInfo,
    dist_info: _DistInfo,
    custom_reset_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]] = None,
) -> DataPipe:
    r"""
    Based on the distributed shared random seed and worker id, this function is used to
    reset the random state of the ``DataPipe`` graph and the global random states for ``torch``,
    ``random`` and ``numpy``.
    """
    # Reset non-sharding process first
    graph = traverse_dps(datapipe)
    non_sharding_process_dps = find_dps(graph, communication.iter._IterateQueueDataPipes)
    if len(non_sharding_process_dps) > 0:
        assert len(non_sharding_process_dps) == 1
        non_sharding_process_dp = non_sharding_process_dps[0]
        # Only send the reset epoch message once
        if worker_info.worker_id == 0:
            non_sharding_reset_fn = partial(non_sharding_process_reset_fn, dist_info=dist_info)
            # Use WorkerInfo(1, 0)
            non_sharding_process_dp.reset_epoch(non_sharding_reset_fn)

    # This function will receive worker local copy of datapipe and reset function from ``initialize_iteration``
    worker_seed_generator = torch.Generator()
    worker_seed_generator.manual_seed(dist_info.shared_seed)
    # TODO(ejguan): https://github.com/pytorch/data/issues/885
    torch.utils.data.graph_settings.apply_random_seed(
        datapipe,
        worker_seed_generator,
    )
    # Set different seeds across distributed workers
    global_worker_id = worker_info.worker_id * dist_info.world_size + dist_info.rank
    worker_seed_generator.manual_seed(dist_info.shared_seed + global_worker_id)

    # Set global random states
    _set_global_random_state(worker_seed_generator)

    if custom_reset_fn is not None:
        datapipe = custom_reset_fn(datapipe, worker_info)
        assert isinstance(datapipe, (IterDataPipe, MapDataPipe))

    return datapipe
