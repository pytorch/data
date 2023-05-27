# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import Callable, Optional

import torch

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from torchdata.dataloader2 import communication
from torchdata.dataloader2.graph import (
    DataPipe,
    find_dps,
    list_dps,
    replace_dp,
    set_datapipes_seed,
    set_graph_random_seed,
    traverse_dps,
)
from torchdata.dataloader2.random import SeedGenerator
from torchdata.dataloader2.utils.dispatch import _DummyIterDataPipe, find_non_dispatching_branches
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


def process_init_fn(
    datapipe: DataPipe,
    worker_info: WorkerInfo,
    custom_init_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]] = None,
    worker_prefetch_cnt: int = 0,
    dispatching_req_queue: Optional[Queue] = None,
    dispatching_res_queue: Optional[Queue] = None,
) -> DataPipe:
    r"""
    Based on the worker information, shard the ``DataPipe`` graph dynamically.
    """
    # Find if there is non-replicable DataPipe
    graph = traverse_dps(datapipe)
    non_replicable_dp = find_dps(graph, _DummyIterDataPipe)  # type: ignore

    # There are two cases for DataPipe graph in terms of mp sharding:
    # 1) All DataPipes are replicable, apply mp sharding to the whole graph
    if len(non_replicable_dp) == 0:
        torch.utils.data.graph_settings.apply_sharding(
            datapipe, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
        )
        assert dispatching_req_queue is None and dispatching_res_queue is None
    # 2) There is non-replicable DataPipe. Since we have replaced the lowest common
    #    ancestor by a `_DummyIterDataPipe`, we would only apply mp sharding
    #    to replicable branches that don't have `_DummyIterDataPipe`.
    else:
        assert len(non_replicable_dp) == 1
        assert not (dispatching_req_queue is None and dispatching_res_queue is None)
        dispatching_req_queue.cancel_join_thread()  # type: ignore[union-attr]
        non_dispatching_branches = find_non_dispatching_branches(graph)
        for dp in non_dispatching_branches:
            torch.utils.data.graph_settings.apply_sharding(
                dp, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
            )

        queue_wrapper = communication.iter.QueueWrapper(
            communication.protocol.IterDataPipeQueueProtocolClient(dispatching_req_queue, dispatching_res_queue)
        )
        dispatch_process_dp = communication.iter._IterateQueueDataPipes([queue_wrapper])
        graph = replace_dp(graph, non_replicable_dp[0], dispatch_process_dp)
        datapipe = list(graph.values())[0][0]

    if custom_init_fn is not None:
        datapipe = custom_init_fn(datapipe, worker_info)
        assert isinstance(datapipe, (IterDataPipe, MapDataPipe))

    if worker_prefetch_cnt > 0:
        datapipe = datapipe.prefetch(worker_prefetch_cnt)

    return datapipe


def _set_global_random_state(seed_generator: SeedGenerator, distributed_shared: bool = False) -> None:
    py_seed = seed_generator.generate_shared_seed() if distributed_shared else seed_generator.generate_seed()
    random.seed(py_seed)

    torch_seed = seed_generator.generate_shared_seed() if distributed_shared else seed_generator.generate_seed()
    torch.manual_seed(torch_seed)

    if HAS_NUMPY:
        # Convert uint64 to uint32 for Numpy
        np_seed = seed_generator.generate_shared_seed() if distributed_shared else seed_generator.generate_seed()
        np_seed = np_seed >> 32
        numpy.random.seed(np_seed)


def process_reset_fn(
    datapipe: DataPipe,
    worker_info: WorkerInfo,
    seed_generator: SeedGenerator,
    distributed_shared_seed: bool = False,
    iter_reset_fn: Optional[Callable[[DataPipe], DataPipe]] = None,
    custom_reset_fn: Optional[Callable[[DataPipe, WorkerInfo, SeedGenerator], DataPipe]] = None,
) -> DataPipe:
    r"""
    Based on the distributed shared random seed and worker id, this function is used to
    reset the random state of the ``DataPipe`` graph and the global random states for ``torch``,
    ``random`` and ``numpy``.
    """
    # Set global random states
    _set_global_random_state(seed_generator, distributed_shared=distributed_shared_seed)

    if distributed_shared_seed:
        graph = traverse_dps(datapipe)
        dps = list_dps(graph)
        set_datapipes_seed(dps, seed_generator=seed_generator, distributed_shared=distributed_shared_seed)
    else:
        set_graph_random_seed(datapipe, seed_generator)

    if iter_reset_fn is not None:
        datapipe = iter_reset_fn(datapipe)
        assert isinstance(datapipe, (IterDataPipe, MapDataPipe))

    if custom_reset_fn is not None:
        datapipe = custom_reset_fn(datapipe, worker_info, seed_generator)
        assert isinstance(datapipe, (IterDataPipe, MapDataPipe))

    return datapipe
