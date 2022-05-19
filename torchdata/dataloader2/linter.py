# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchdata.datapipes.iter import IterDataPipe, ShardingFilter, Shuffler

from ._graph_utils import DataPipeGraph, traverse


def _check_shuffle_before_sharding(datapipe: IterDataPipe) -> bool:
    """
    This function will check if a ``shuffle`` operation is presented before each
    ``sharding_filter`` operation for every single path in the ``DataPipe`` graph.
    """
    graph: DataPipeGraph = traverse(datapipe)
    return _check_shuffler_before_sharding_helper(graph)


def _check_shuffler_before_sharding_helper(graph: DataPipeGraph) -> bool:
    if not graph:
        return True

    if len(graph) > 1:
        for dp in graph:
            if isinstance(dp, ShardingFilter):
                if not _has_shuffler(graph[dp]):
                    return False
            else:
                if not _check_shuffler_before_sharding_helper(graph[dp]):
                    return False
        return True

    dp, dp_graph = list(graph.items())[0]
    if isinstance(dp, ShardingFilter):
        return _has_shuffler(dp_graph)

    return _check_shuffler_before_sharding_helper(dp_graph)


def _has_shuffler(graph: DataPipeGraph) -> bool:
    if not graph:
        return False

    if len(graph) > 1:
        for dp in graph:
            if not (isinstance(dp, Shuffler) or _has_shuffler(graph[dp])):
                return False
        return True

    dp, dp_graph = list(graph.items())[0]
    if isinstance(dp, Shuffler):
        return True

    return _has_shuffler(dp_graph)
