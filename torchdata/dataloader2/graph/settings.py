# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import inspect

import torch

from torchdata.dataloader2.graph import DataPipe, find_dps, list_dps, traverse_dps
from torchdata.dataloader2.utils import generate_random_int
from torchdata.datapipes.iter import ShardingFilter


def _is_random_datapipe(datapipe: DataPipe) -> bool:
    if hasattr(datapipe, "set_seed") and inspect.ismethod(datapipe.set_seed):
        return True
    return False


def _seed_datapipe(datapipe: DataPipe, seed_generator: torch.Generator) -> DataPipe:
    seed = generate_random_int(seed_generator)
    datapipe.set_seed(seed)
    return datapipe


def _set_worker_seed_for_dp_graph(datapipe: DataPipe, seed_generator: torch.Generator, worker_id: int = 0) -> DataPipe:
    r"""
    Set seeds to the graph of ``DataPipes`` based on a Seed Generator. All random ``DataPipes`` prior to
    ``ShardingFilter`` will be set seeds by the same Seed Generator to preserve the same random state
    across distributed/non-distributed workers. And, the random ``DataPipes`` after ``ShardingFilter``
    will be set seeds by the worker-local Seed Generator deterministically created based on ``worker_id``.
    """
    graph = traverse_dps(datapipe)
    sharding_filter_dps = find_dps(graph, ShardingFilter)

    base_seed = generate_random_int(seed_generator)
    worker_seed = base_seed + worker_id

    # Set the same seed before sharding_filter
    # Using cache to exclude potential duplciate DataPipe
    cache = set()
    dps_before_sharding = []
    for sf_dp in sharding_filter_dps:
        dps = list_dps(traverse_dps(sf_dp))
        for dp in dps:
            if id(dp) not in cache:
                cache.add(id(dp))
                dps_before_sharding.append(dp)

    for dp in dps_before_sharding:
        if _is_random_datapipe(dp):
            _seed_datapipe(dp, seed_generator)

    # Set different seeds after sharding_filter
    seed_generator.manual_seed(worker_seed)
    dps = list_dps(graph, exclude_dps=sharding_filter_dps)
    for dp in dps:
        if _is_random_datapipe(dp):
            _seed_datapipe(dp, seed_generator)

    return datapipe
