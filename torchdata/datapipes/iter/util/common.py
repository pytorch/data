# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, TypeVar

from torchdata.datapipes.iter import FlatMapper, Mapper

T_co = TypeVar("T_co", covariant=True)


class MapTemplateIterDataPipe(Mapper[T_co]):
    def __init__(self, source_datapipe, input_col=None, output_col=None) -> None:
        fn = self._map
        super().__init__(source_datapipe, fn=fn, input_col=input_col, output_col=output_col)

    def _map(self, *args, **kwargs) -> T_co:
        raise NotImplementedError


class FlatMapTemplateIterDataPipe(FlatMapper[T_co]):
    def __init__(self, source_datapipe, input_col=None, output_col=None, length: int = -1) -> None:
        fn = self._flatmap
        super().__init__(source_datapipe, fn=fn, input_col=input_col, output_col=output_col, length=length)

    def _flatmap(self, *args, **kwargs) -> Iterable[T_co]:
        raise NotImplementedError
