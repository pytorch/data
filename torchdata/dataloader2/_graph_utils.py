# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
import pickle
from typing import Dict, List

from torch.utils.data import IterDataPipe

DataPipeGraph = Dict[IterDataPipe, "DataPipeGraph"]


# Modified based on torch.utils.data.graph
def stub_unpickler() -> str:
    return "STUB"


def list_connected_datapipes(scan_obj: IterDataPipe, only_datapipe: bool) -> List[IterDataPipe]:

    f = io.BytesIO()
    p = pickle.Pickler(f)  # Not going to work for lambdas, but dill infinite loops on typing and can't be used as is

    captured_connections: List[IterDataPipe] = []

    def getstate_hook(obj: IterDataPipe) -> Dict[str, IterDataPipe]:
        state = {}
        for k, v in obj.__dict__.items():
            if isinstance(v, (IterDataPipe, dict, list, set, tuple)):  # including all potential containers
                state[k] = v
        return state

    def reduce_hook(obj: IterDataPipe):  # pyre-ignore
        if obj == scan_obj:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            return stub_unpickler, ()

    try:
        IterDataPipe.set_reduce_ex_hook(reduce_hook)
        if only_datapipe:
            IterDataPipe.set_getstate_hook(getstate_hook)
        p.dump(scan_obj)
    except AttributeError:  # unpickable DataPipesGraph
        pass  # TODO(VitalyFedyunin): We need to tight this requirement after migrating from old DataLoader
    finally:
        IterDataPipe.set_reduce_ex_hook(None)
        if only_datapipe:
            IterDataPipe.set_getstate_hook(None)
    return captured_connections


def traverse(datapipe: IterDataPipe, only_datapipe: bool = False) -> DataPipeGraph:
    if not isinstance(datapipe, IterDataPipe):
        raise RuntimeError(f"Expected `IterDataPipe`, but {type(datapipe)} is found")

    items: List[IterDataPipe] = list_connected_datapipes(datapipe, only_datapipe)
    d = {datapipe: {}}
    for item in items:
        d[datapipe].update(traverse(item, only_datapipe))
    return d
