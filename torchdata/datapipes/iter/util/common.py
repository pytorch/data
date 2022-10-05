# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import TypeVar

from torchdata.datapipes.iter import FlatMapperProto, Mapper

T_co = TypeVar("T_co", covariant=True)


class MapTemplateIterDataPipe(Mapper[T_co]):
    def __init__(self, source_datapipe, input_col=None, output_col=None) -> None:
        fn = getattr(self, "_map")
        assert fn is not None
        super().__init__(source_datapipe, fn=fn, input_col=input_col, output_col=output_col)

    # TODO(VitalyFedyunin): MyPy doesn't allow disabling of `incompatible with supertype` for child class
    # def _map(self) -> T_co:
    #    raise NotImplementedError


class FlatMapTemplateIterDataPipe(FlatMapperProto[T_co]):
    def __init__(self, source_datapipe, input_col=None, output_col=None, length: int = -1) -> None:
        fn = getattr(self, "_flatmap")
        assert fn is not None
        super().__init__(source_datapipe, fn=fn, input_col=input_col, output_col=output_col, length=length)

    # TODO(VitalyFedyunin): MyPy doesn't allow disabling of `incompatible with supertype` for child class
    # def _flatmap(self) -> Iterable[T_co]:
    #     raise NotImplementedError
