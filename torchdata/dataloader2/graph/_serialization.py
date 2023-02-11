# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pickle

from torch.utils.data.datapipes.datapipe import (
    _DataPipeSerializationWrapper,
    _IterDataPipeSerializationWrapper,
    _MapDataPipeSerializationWrapper,
)

from torchdata.dataloader2.graph import DataPipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe

try:
    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

__all__ = [
    "attach_wrapper",
    "clone",
    "deserialize_datapipe",
    "extract_wrapper",
    "serialize_datapipe",
]


def serialize_datapipe(datapipe: DataPipe) -> bytes:
    datapipe = attach_wrapper(datapipe)
    try:
        return pickle.dumps(datapipe)
    except pickle.PickleError as e:
        raise NotImplementedError(f"Prototype only support pickle-able datapipes for checkpoint: {e}")


def deserialize_datapipe(serialized_state: bytes) -> DataPipe:
    try:
        datapipe = pickle.loads(serialized_state)
    except pickle.PickleError as e:
        raise NotImplementedError(f"Prototype only support pickle-able datapipes for checkpoint: {e}")
    return extract_wrapper(datapipe)


def attach_wrapper(datapipe: DataPipe) -> DataPipe:
    r"""
    Wraps the ``DataPipe`` with the corresponding serialization wrapper.
    """
    wrapped_dp: DataPipe = datapipe
    if not isinstance(datapipe, _DataPipeSerializationWrapper):
        if isinstance(datapipe, IterDataPipe):
            wrapped_dp = _IterDataPipeSerializationWrapper(datapipe)
        elif isinstance(datapipe, MapDataPipe):
            wrapped_dp = _MapDataPipeSerializationWrapper(datapipe)
    return wrapped_dp


def extract_wrapper(datapipe: DataPipe) -> DataPipe:
    r"""
    Extracts the ``DataPipe`` from the serialization wrapper.
    """
    if isinstance(datapipe, _DataPipeSerializationWrapper):
        datapipe = datapipe._datapipe
    return datapipe


def clone(obj):
    r"""
    Standardized way to copy an object when needed, such as for DataPipe/ReadingService.
    This uses `pickle` to serialize/deserialize to create the copy.
    """
    use_dill = False
    try:
        states = pickle.dumps(obj)
    except Exception:
        if HAS_DILL:
            states = dill.dumps(obj)
            use_dill = True
        else:
            raise
    if use_dill:
        return dill.loads(states)
    else:
        return pickle.loads(states)
