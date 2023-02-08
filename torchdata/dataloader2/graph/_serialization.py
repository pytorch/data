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
from torchdata.dataloader2.random.seed_generator import SeedGenerator
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe


__all__ = [
    "clone",
    "deserialize_datapipe",
    "serialize_datapipe",
    "serialize_seed_generator",
    "wrap_datapipe_for_serialization",
]


def serialize_seed_generator(seed_generator: SeedGenerator) -> bytes:
    try:
        return pickle.dumps(seed_generator)
    except pickle.PickleError as e:
        raise RuntimeError(f"Seed generator should be pickle-able by default for checkpoint: {e}")


def deserialize_seed_generator(serialized_generator: bytes) -> SeedGenerator:
    try:
        return pickle.loads(serialized_generator)
    except pickle.PickleError as e:
        raise RuntimeError(f"Seed generator should be pickle-able by default for checkpoint: {e}")


def serialize_datapipe(datapipe: DataPipe) -> bytes:
    try:
        return pickle.dumps(datapipe)
    except pickle.PickleError as e:
        raise NotImplementedError(f"Prototype only support pickle-able datapipes for checkpoint: {e}")


def deserialize_datapipe(serialized_state: bytes) -> DataPipe:
    try:
        return pickle.loads(serialized_state)
    except pickle.PickleError as e:
        raise NotImplementedError(f"Prototype only support pickle-able datapipes for checkpoint: {e}")


def wrap_datapipe_for_serialization(datapipe: DataPipe):
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


def clone(obj):
    r"""
    Standardized way to copy an object when needed, such as for DataPipe/ReadingService.
    This uses `pickle` to serialize/deserialize to create the copy.
    """
    return pickle.loads(pickle.dumps(obj))
