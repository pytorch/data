# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from typing import Callable, Iterable, Iterator, TypeVar

from torch.utils.data.datapipes.utils.common import _check_unpickable_fn
from torchdata.datapipes import functional_datapipe

from torchdata.datapipes.iter import IterDataPipe, Mapper

T_co = TypeVar("T_co", covariant=True)


class MapTemplateIterDataPipe(Mapper[T_co]):
    def __init__(self, source_datapipe, input_col=None, output_col=None) -> None:
        fn = self._map
        super().__init__(source_datapipe, fn=fn, input_col=input_col, output_col=output_col)

    def _map(self, *args, **kwargs) -> T_co:
        raise NotImplementedError


@functional_datapipe("flatmap_proto")
class FlatMapperProtoIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable,
        input_col=None,
        output_col=None,
        length: int = -1,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.length = length

        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

        self.input_col = input_col
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        self.output_col = output_col

    def _apply_fn(self, data):
        # TODO(VitalyFedyunin): Can be unified with _apply_fn of MapperProtoIterDataPipe
        if self.input_col is None and self.output_col is None:
            yield from self.fn(data)
            return

        if self.input_col is None:
            res = self.fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            res = self.fn(*args)
        else:
            res = self.fn(data[self.input_col])

        # Copy tuple to list and run in-place modification because tuple is immutable.
        if isinstance(data, tuple):
            t_flag = True
            data = list(data)
        else:
            t_flag = False

        for res_line in res:
            data_copy = copy.deepcopy(data)
            if self.output_col is None:
                if isinstance(self.input_col, (list, tuple)):
                    data_copy[self.input_col[0]] = res_line
                    for idx in sorted(self.input_col[1:], reverse=True):
                        del data_copy[idx]
                else:
                    data_copy[self.input_col] = res_line
            else:
                if self.output_col == -1:
                    data_copy.append(res_line)
                else:
                    data_copy[self.output_col] = res_line

            # Convert list back to tuple
            yield tuple(data_copy) if t_flag else data_copy

    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            yield from self._apply_fn(data)

    def __len__(self) -> int:
        if self.length != -1:
            return self.length
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


class FlatMapTemplateIterDataPipe(FlatMapperProtoIterDataPipe[T_co]):
    def __init__(self, source_datapipe, input_col=None, output_col=None, length: int = -1) -> None:
        fn = self._flatmap
        super().__init__(source_datapipe, fn=fn, input_col=input_col, output_col=output_col, length=length)

    def _flatmap(self, *args, **kwargs) -> Iterable[T_co]:
        raise NotImplementedError
