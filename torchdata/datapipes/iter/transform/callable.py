# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Callable, Hashable, Iterator, List, Optional, Sized, TypeVar, Union

from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

T_co = TypeVar("T_co", covariant=True)


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

        _check_unpickable_fn(fn)
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

        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]
        self.input_col = input_col

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


@functional_datapipe("drop")
class DropperIterDataPipe(IterDataPipe[T_co]):
    r"""
    Drop columns/elements in input DataPipe via its indices (functional name: ``drop``).

    Args:
        datapipe: IterDataPipe with columns to be dropped
        indices: a single column index to be dropped or a list of indices

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, ZipperMapDataPipe
        >>> dp1 = IterableWrapper(range(5))
        >>> dp2 = IterableWrapper(range(10, 15))
        >>> dp = dp1.zip(dp2)
        >>> list(dp)
        [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]
        >>> drop_dp = dp.drop(1)
        >>> list(drop_dp)
        [(0), (1), (2), (3), (4)]
    """
    datapipe: IterDataPipe

    def __init__(
        self,
        datapipe: IterDataPipe,
        indices: Union[Hashable, List[Hashable]],
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        if isinstance(indices, list):
            self.indices = set(indices)
        else:
            self.indices = {indices}

    def __iter__(self) -> Iterator[T_co]:
        for old_item in self.datapipe:
            if isinstance(old_item, tuple):
                new_item = tuple(x for i, x in enumerate(old_item) if i not in self.indices)  # type: ignore[assignment]
            elif isinstance(old_item, list):
                new_item = [x for i, x in enumerate(old_item) if i not in self.indices]  # type: ignore[assignment]
            elif isinstance(old_item, dict):
                new_item = {k: v for (k, v) in old_item.items() if k not in self.indices}  # type: ignore[assignment]
            else:
                new_item = old_item
                warnings.warn(
                    "The next item was not an iterable and cannot be filtered, "
                    "please be aware that no filter was done or new item created."
                )

            # check to make sure all indices requested were in the item. warn if not
            try:
                for i in self.indices:
                    old_item[i]
            except (IndexError, KeyError):
                warnings.warn(
                    "At least one index in the filter is not present in the item being returned,"
                    " please be aware that expected columns/keys may be missing."
                )

            yield new_item  # type: ignore[misc]

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe("islice")
class ISliceIterDataPipe(IterDataPipe[T_co]):
    r"""
    returns a slice of elements in input DataPipe via start/stop/step or indices (functional name: ``islice``).

    Args:
        datapipe: IterDataPipe with iterable elements
        index: a single start index for the slice or a list of indices to be returned instead of a start/stop slice
        stop: the slice stop. ignored if index is a list
        step: step to be taken from start to stop. ignored if index is a list

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper([(0, 10, 100), (1, 11, 111), (2, 12, 122), (3, 13, 133), (4, 14, 144)])
        >>> islice_dp = dp.islice(0, 2)
        >>> list(islice_dp)
        [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]
    """
    datapipe: IterDataPipe

    def __init__(
        self,
        datapipe: IterDataPipe,
        index: Union[int, List[Hashable]],
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe

        self.index = index
        self.stop = stop
        self.step = step

    def __iter__(self) -> Iterator[T_co]:
        for old_item in self.datapipe:
            if isinstance(old_item, tuple):
                if isinstance(self.index, list):
                    new_item = tuple(x for i, x in enumerate(old_item) if i in self.index)  # type: ignore[assignment]
                else:
                    new_item = old_item[self.index : self.stop : self.step]  # type: ignore[assignment]
            elif isinstance(old_item, list):
                if isinstance(self.index, list):
                    new_item = [x for i, x in enumerate(old_item) if i in self.index]  # type: ignore[assignment]
                else:
                    new_item = old_item[self.index : self.stop : self.step]  # type: ignore[assignment]
            elif isinstance(old_item, dict):
                if isinstance(self.index, list):
                    new_item = {k: v for (k, v) in old_item.items() if k in self.index}  # type: ignore[assignment]
                else:
                    new_keys = list(old_item.keys())[self.index : self.stop : self.step]
                    new_item = {k: v for (k, v) in old_item.items() if k in new_keys}  # type: ignore[assignment]
            else:
                new_item = old_item
                warnings.warn(
                    "The next item was not an iterable and cannot be filtered, "
                    "please be aware that no filter was done or new item created."
                )

            if isinstance(self.index, list):
                # check to make sure all indices requested were in the item. warn if not
                try:
                    for i in self.index:
                        old_item[i]
                except (IndexError, KeyError):
                    warnings.warn(
                        "At least one index in the filter is not present in the item being returned,"
                        " please be aware that expected columns/keys may be missing."
                    )

            yield new_item  # type: ignore[misc]

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
