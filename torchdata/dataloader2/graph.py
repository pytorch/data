# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Type

from ._graph_utils import DataPipe, DataPipeGraph, traverse


# In case that there will be multiple datapipe needs to be adapted
def find_dps(graph: DataPipeGraph, dp_type: Type[DataPipe]) -> List[DataPipe]:
    dps: List[DataPipe] = []

    def helper(g) -> None:  # pyre-ignore
        for dp in g:
            if type(dp) is dp_type:  # Please not use `isinstance`, there is a bug.
                dps.append(dp)
            src_graph = g[dp]
            helper(src_graph)

    helper(graph)

    return dps


# Given the DataPipe needs to be adapted and the expected DataPipe, return a new graph
def replace_dp(graph: DataPipeGraph, old_datapipe: DataPipe, new_datapipe: DataPipe) -> DataPipeGraph:
    if old_datapipe is new_datapipe:
        return graph
    if old_datapipe in graph:
        g = graph.pop(old_datapipe)
        graph[new_datapipe] = g
    for recv_dp in graph:
        _replace_dp(recv_dp, graph[recv_dp], old_datapipe, new_datapipe)

    # Get the last DataPipe in graph
    datapipe = list(graph.keys())[0]

    return traverse(datapipe, only_datapipe=True)


# Given the DataPipe needs to be removed, return a new graph
def remove_dp(graph: DataPipeGraph, datapipe: DataPipe) -> DataPipeGraph:
    if datapipe in graph:
        graph = graph[datapipe]
        if len(graph) == 0:
            raise Exception("Cannot remove source DataPipe that is the first DataPipe in the pipeline")
        if len(graph) > 1:
            raise Exception("Cannot remove a receiving DataPipe having multiple sending DataPipes")
    for recv_dp in graph:
        _remove_dp(recv_dp, graph[recv_dp], datapipe)

    # Get the last DataPipe in graph
    datapipe = list(graph.keys())[0]

    return traverse(datapipe, only_datapipe=True)


# For each `recv_dp`, find if the source_datapipe needs to be replaced by the new one.
# If found, find where the `old_dp` is located in `dp` and switch it to the `new_dp`
def _remove_dp(recv_dp, send_graph: DataPipeGraph, datapipe: DataPipe) -> None:
    for send_dp in send_graph:
        if send_dp is datapipe:
            g = send_graph[send_dp]
            if len(g) == 0:
                raise Exception("Cannot remove source DataPipe that is the first DataPipe in the pipeline")
            if len(g) > 1:
                raise Exception("Cannot remove a receiving DataPipe having multiple sending DataPipes")
            src_dp = list(g.keys())[0]
            _assign_attr(recv_dp, send_dp, src_dp, inner_dp=True)
        else:
            _remove_dp(send_dp, send_graph[send_dp], datapipe)


# For each `recv_dp`, find if the source_datapipe needs to be replaced by the new one.
# If found, find where the `old_dp` is located in `recv_dp` and switch it to the `new_dp`
def _replace_dp(recv_dp, send_graph: DataPipeGraph, old_dp: DataPipe, new_dp: DataPipe) -> None:
    for send_dp in send_graph:
        if send_dp is old_dp:
            _assign_attr(recv_dp, old_dp, new_dp, inner_dp=True)
        else:
            _replace_dp(send_dp, send_graph[send_dp], old_dp, new_dp)


# Recursively re-assign datapipe for the sake of nested data structure
# `inner_dp` is used to prevent recursive call if we have already met `DataPipe`
def _assign_attr(obj, old_dp, new_dp, inner_dp: bool = False):
    if obj is old_dp:
        return new_dp
    elif isinstance(obj, DataPipe):
        # Prevent recursive call for DataPipe
        if not inner_dp:
            return None
        for k in list(obj.__dict__.keys()):
            new_obj = _assign_attr(obj.__dict__[k], old_dp, new_dp)
            if new_obj is not None:
                obj.__dict__[k] = new_obj
                break
        return None
    elif isinstance(obj, dict):
        for k in list(obj.keys()):
            new_obj = _assign_attr(obj[k], old_dp, new_dp)
            if new_obj is not None:
                obj[k] = new_obj
                break
        return None
    # Tuple is immutable, has to re-create a tuple
    elif isinstance(obj, tuple):
        temp_list = []
        flag = False
        for o in obj:
            new_obj = _assign_attr(o, old_dp, new_dp, inner_dp)
            if new_obj is not None:
                flag = True
                temp_list.append(new_dp)
            else:
                temp_list.append(o)
        if flag:
            return tuple(temp_list)  # Special case
        else:
            return None
    elif isinstance(obj, list):
        for i in range(len(obj)):
            new_obj = _assign_attr(obj[i], old_dp, new_dp, inner_dp)
            if new_obj is not None:
                obj[i] = new_obj
                break
        return None
    elif isinstance(obj, set):
        new_obj = None
        for o in obj:
            if _assign_attr(o, old_dp, new_dp, inner_dp) is not None:
                new_obj = new_dp
                break
        if new_obj is not None:
            obj.remove(old_dp)
            obj.add(new_dp)
        return None
    else:
        return None
