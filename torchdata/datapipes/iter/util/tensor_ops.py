# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

from typing import Dict, Iterable, Iterator, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


def capture_source(func):
    @functools.wraps(func)
    def wrapper_capture_source(self, source_datapipe, *args, **kwargs):
        self.source_datapipe = source_datapipe
        return func(self, source_datapipe, *args, **kwargs)

    return wrapper_capture_source


@functional_datapipe("pin_memory")
class PinMemoryIterDataPipe(IterDataPipe[T_co]):
    """
    TODO:
    """

    @capture_source
    def __init__(self, source_datapipe: IterDataPipe[T_co]) -> None:
        pass

    @staticmethod
    def pin_or_raise(object):
        try:
            # DuckTyping here, as it might be any Tensor extention class
            return object.pin_memory()
        except AttributeError:
            raise Exception("Ubable to pin memory")

    @staticmethod
    def pin_memory(object):
        if isinstance(object, Dict):
            dict_result = {key: PinMemoryIterDataPipe.pin_or_raise(value) for key, value in object.items()}
            return dict_result
        elif isinstance(object, Iterable):
            list_result = [PinMemoryIterDataPipe.pin_or_raise(value) for value in object]
            return list_result
        else:
            return PinMemoryIterDataPipe.pin_or_raise(object)

    def __iter__(self) -> Iterator[T_co]:
        for item in self.source_datapipe:
            yield self.pin_memory(item)

    def __len__(self) -> int:
        return len(self.source_datapipe)
