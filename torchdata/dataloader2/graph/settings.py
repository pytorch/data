# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect

from typing import List

from torchdata.dataloader2.graph.utils import DataPipe, find_dps, list_dps, traverse_dps
from torchdata.dataloader2.random import SeedGenerator
from torchdata.datapipes.iter import ShardingFilter


def _is_random_datapipe(datapipe: DataPipe) -> bool:
    if hasattr(datapipe, "set_seed") and inspect.ismethod(datapipe.set_seed):
        return True
    return False


def set_datapipes_seed(datapipes: List[DataPipe], seed_generator: SeedGenerator, distributed_shared: bool) -> None:
    for dp in datapipes:
        if _is_random_datapipe(dp):
            if distributed_shared:
                dp.set_seed(seed_generator.generate_shared_seed())
            else:
                dp.set_seed(seed_generator.generate_seed())


def set_graph_random_seed(datapipe: DataPipe, seed_generator: SeedGenerator) -> DataPipe:
    r"""
    Set seeds to the graph of ``DataPipes`` based on a Seed Generator. All random ``DataPipes`` prior to
    ``ShardingFilter`` will be set seeds by the same Seed Generator to preserve the same random state
    across distributed/non-distributed workers. And, the random ``DataPipes`` after ``ShardingFilter``
    will be set seeds by the worker-local Seed Generator deterministically created based on ``worker_id``.

    Args:
        datapipe:
        seed_generator:
    """
    graph = traverse_dps(datapipe)
    sharding_filter_dps = find_dps(graph, ShardingFilter)

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

    set_datapipes_seed(dps_before_sharding, seed_generator, distributed_shared=True)

    # Set different seeds after sharding_filter
    dps_after_sharding = list_dps(graph, exclude_dps=sharding_filter_dps)
    set_datapipes_seed(dps_after_sharding, seed_generator, distributed_shared=False)

    return datapipe
