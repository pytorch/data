# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata.dataloader2.graph import DataPipe, DataPipeGraph, traverse_dps

from torchdata.datapipes.iter import ShardingFilter, Shuffler


def _check_shuffle_before_sharding(datapipe: DataPipe) -> bool:
    """
    This function will check if a ``shuffle`` operation is presented before each
    ``sharding_filter`` operation for every single path in the ``DataPipe`` graph.
    """
    graph: DataPipeGraph = traverse_dps(datapipe)  # type: ignore[arg-type]
    return _check_shuffler_before_sharding_helper(graph)


def _check_shuffler_before_sharding_helper(graph: DataPipeGraph) -> bool:
    if not graph:
        return True

    if len(graph) > 1:
        for dp, sub_graph in graph.values():
            if isinstance(dp, ShardingFilter):
                if not _has_shuffler(sub_graph):
                    return False
            else:
                if not _check_shuffler_before_sharding_helper(sub_graph):
                    return False
        return True

    dp, dp_graph = list(graph.values())[0]
    if isinstance(dp, ShardingFilter):
        return _has_shuffler(dp_graph)

    return _check_shuffler_before_sharding_helper(dp_graph)


def _has_shuffler(graph: DataPipeGraph) -> bool:
    if not graph:
        return False

    if len(graph) > 1:
        for dp, sub_graph in graph.values():
            if not (isinstance(dp, Shuffler) or _has_shuffler(sub_graph)):
                return False
        return True

    dp, dp_graph = list(graph.values())[0]
    if isinstance(dp, Shuffler):
        return True

    return _has_shuffler(dp_graph)
