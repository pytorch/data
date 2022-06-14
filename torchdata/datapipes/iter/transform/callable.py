# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Callable, Iterator, List, Optional, TypeVar, Union

from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_lambda_fn

T_co = TypeVar("T_co", covariant=True)


def ensure_fn_works(fn: Callable, input_col: Optional[Union[int, tuple, list]]):
    """
    Checks that function is compatible with the input column.

    Args:
        fn: The function to check.
        input_col: The input column to check.

    Returns:
        None.

    Raises:
        TypeError: If the function is not compatible with the input column.
    """
    sig = inspect.signature(fn)

    if isinstance(input_col, int) or not input_col:
        if len(sig.parameters) != 1:
            raise TypeError(
                f"The function {fn.__name__} takes {len(sig.parameters)} " f"arguments, " f"but 1 is required."
            )
    elif isinstance(input_col, (list, tuple)):
        if len(sig.parameters) != len(input_col):
            raise TypeError(
                f"The function {fn.__name__} takes {len(sig.parameters)} "
                f"arguments, "
                f"but {len(input_col)} are required."
            )


@functional_datapipe("map_batches")
class BatchMapperIterDataPipe(IterDataPipe[T_co]):
    r"""
    Combines elements from the source DataPipe to batches and applies a function
    over each batch, then flattens the outpus to a single, unnested IterDataPipe
    (functional name: ``map_batches``).

    Args:
        datapipe: Source IterDataPipe
        fn: The function to be applied to each batch of data
        batch_size: The size of batch to be aggregated from ``datapipe``
        input_col: Index or indices of data which ``fn`` is applied, such as:
            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def fn(batch):
        >>>     return [d + 1 for d in batch]
        >>> source_dp = IterableWrapper(list(range(5)))
        >>> mapped_dp = source_dp.map_batches(fn, batch_size=3)
        >>> list(mapped_dp)
        [1, 2, 3, 4, 5]

    Notes:
        Compared with ``map``, the reason that ``map_batches`` doesn't take
        ``output_col`` argument is the size of ``fn`` output is not guaranteed
        to be the same as input batch. With different size, this operation cannot
        assign data back to original data structure.

        And, this operation is introduced based on the use case from `TorchText`.
        A pybinded C++ vectorized function can be applied for efficiency.
    """
    datapipe: IterDataPipe
    fn: Callable
    batch_size: int

    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable,
        batch_size: int,
        input_col=None,
    ) -> None:
        self.datapipe = datapipe

        _check_lambda_fn(fn)
        self.fn = fn  # type: ignore[assignment]

        assert batch_size > 0, "Batch size is required to be larger than 0!"
        self.batch_size = batch_size
        self.input_col = input_col

    def _apply_fn(self, batch):
        if self.input_col is None:
            return self.fn(batch)

        if isinstance(self.input_col, (list, tuple)):
            args = [[data[idx] for idx in self.input_col] for data in batch]
        else:
            args = [data[self.input_col] for data in batch]
        return self.fn(args)

    def __iter__(self) -> Iterator[T_co]:
        batch: List = []
        for d in self.datapipe:
            batch.append(d)
            if len(batch) == self.batch_size:
                yield from self._apply_fn(batch)
                batch = []
        if batch:
            yield from self._apply_fn(batch)

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__}'s length relies on the output of its function.")


@functional_datapipe("flatmap")
class FlatMapperIterDataPipe(IterDataPipe[T_co]):
    r"""
    Applies a function over each item from the source DataPipe, then
    flattens the outputs to a single, unnested IterDataPipe (functional name: ``flatmap``).

    Note:
        The output from ``fn`` must be a Sequence. Otherwise, an error will be raised.

    Args:
        datapipe: Source IterDataPipe
        fn: the function to be applied to each element in the DataPipe, the output must be a Sequence
        input_col: Index or indices of data which ``fn`` is applied, such as:
            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.


    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def fn(e):
        >>>     return [e, e * 10]
        >>> source_dp = IterableWrapper(list(range(5)))
        >>> flatmapped_dp = source_dp.flatmap(fn)
        >>> list(flatmapped_dp)
        [0, 0, 1, 10, 2, 20, 3, 30, 4, 40]
    """
    datapipe: IterDataPipe
    fn: Callable

    def __init__(self, datapipe: IterDataPipe, fn: Callable, input_col=None) -> None:
        self.datapipe = datapipe

        _check_lambda_fn(fn)
        self.fn = fn  # type: ignore[assignment]
        self.input_col = input_col
        ensure_fn_works(fn, input_col)

    def _apply_fn(self, data):
        if self.input_col is None:
            return self.fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            return self.fn(*args)
        else:
            return self.fn(data[self.input_col])

    def __iter__(self) -> Iterator[T_co]:
        for d in self.datapipe:
            yield from self._apply_fn(d)

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__}'s length relies on the output of its function.")
