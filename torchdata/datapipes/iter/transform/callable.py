# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import inspect
import random
import warnings
from collections import deque
from concurrent import futures

from typing import Callable, Hashable, Iterator, List, Optional, Set, Sized, TypeVar, Union

import torch
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn, validate_input_col
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


def _no_op_fn(*args):
    """
    No-operation function, returns passed arguments.
    """
    if len(args) == 1:
        return args[0]
    return args


@functional_datapipe("map_batches")
class BatchMapperIterDataPipe(IterDataPipe[T_co]):
    r"""
    Combines elements from the source DataPipe to batches and applies a function
    over each batch, then flattens the outputs to a single, unnested IterDataPipe
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
        If ``fn`` is ``None``, source DataPipe will be just flattened vertically, provided that items can be unpacked.

    Args:
        datapipe: Source IterDataPipe
        fn: the function to be applied to each element in the DataPipe, the output must be a Sequence
        input_col: Index or indices of data which ``fn`` is applied, such as:

            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is/are used for list/tuple.
            - Key(s) is/are used for dict.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def fn(e):
        >>>     return [e, e * 10]
        >>> source_dp = IterableWrapper(list(range(5)))
        >>> flatmapped_dp = source_dp.flatmap(fn)
        >>> list(flatmapped_dp)
        [0, 0, 1, 10, 2, 20, 3, 30, 4, 40]
        >>>
        >>> source_dp = IterableWrapper([[1, 2, 3], [4, 5, 6]])
        >>> flatmapped_dp = source_dp.flatmap()
        >>> list(flatmapped_dp)
        [1, 2, 3, 4, 5, 6]
    """
    datapipe: IterDataPipe
    fn: Optional[Callable]

    def __init__(self, datapipe: IterDataPipe, fn: Optional[Callable] = None, input_col=None) -> None:
        self.datapipe = datapipe

        if fn is None:
            fn = _no_op_fn
        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]
        self.input_col = input_col
        validate_input_col(fn, input_col)

    def _apply_fn(self, data):
        if self.input_col is None:
            return self.fn(data)  # type: ignore[misc]
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            return self.fn(*args)  # type: ignore[misc]
        else:
            return self.fn(data[self.input_col])  # type: ignore[misc]

    def __iter__(self) -> Iterator[T_co]:
        for d in self.datapipe:
            yield from self._apply_fn(d)

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__}'s length relies on the output of its function.")


@functional_datapipe("shuffled_flatmap")
class ShuffledFlatMapperIterDataPipe(IterDataPipe):
    r"""
    Applies a function over each item from the source DataPipe,
    then collects the iterables returned in a buffer,
    then, at every iteration, chooses at random one of the iterables in the buffer
    and yields one item from this iterable (functional name: ``shuffled_flatmap``).

    When the buffer is full, the DataPipe will begin to yield elements from iterables within the buffer.
    New iterables will be added to the buffer once the existing ones run out of elements.
    Note:
        The output from ``fn`` must be an Iterable. Otherwise, an error will be raised.
        If ``fn`` is ``None``, source DataPipe will be just flattened vertically, provided that items can be unpacked.

    Args:
        datapipe: Source IterDataPipe
        fn: the function to be applied to each element in the DataPipe, the output must be a Sequence
        input_col: Index or indices of data which ``fn`` is applied, such as:

            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is/are used for list/tuple.
            - Key(s) is/are used for dict.
        buffer_size: the max number of iterables this DataPipe can hold at a time (default to ``100``)

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([[1, 2, 3, 4], 'abcd', 'ABCD'])
        >>> shuffled_flatmapped_dp = source_dp.shuffled_flatmap(buffer_size=2)
        >>> list(shuffled_flatmapped_dp)
        ['a', 'b', 'c', 1, 'd', 'A', 'B', 'C', 2, 'D', 3, 4]
        >>>
        >>> # To shuffle all the elements, you can combine `shuffled_flatmap` with `in_batch_shuffle` like this:
        >>> fully_shuffled_flatmapped_dp = source_dp.in_batch_shuffle()
        >>> fully_shuffled_flatmapped_dp = fully_shuffled_flatmapped_dp.shuffled_flatmap()
        >>> list(fully_shuffled_flatmapped_dp)
        ['b', 3, 'c', 'd', 'C', 'A', 'a', 2, 'B', 'D', 4, 1]
    """
    datapipe: IterDataPipe
    fn: Optional[Callable]
    buffer_size: int
    _buffer: List[Iterator]
    _enabled: bool
    _seed: Optional[int]
    _rng: random.Random
    _no_op_fn: bool = False

    def __init__(
        self, datapipe: IterDataPipe, fn: Optional[Callable] = None, input_col=None, buffer_size: int = 100
    ) -> None:
        super().__init__()
        self._buffer = []
        self.datapipe = datapipe

        if fn is None:
            fn = _no_op_fn
            self._no_op_fn = True
        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]
        self.input_col = input_col
        validate_input_col(fn, input_col)

        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.buffer_size = buffer_size
        self._enabled = True
        self._seed = None
        self._rng = random.Random()

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def reset(self) -> None:
        self._buffer = []
        if self._enabled:
            if self._seed is None:
                self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self._rng.seed(self._seed)
            self._seed = None

    def _apply_fn(self, data):
        if self.input_col is None:
            return self.fn(data)  # type: ignore[misc]
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            return self.fn(*args)  # type: ignore[misc]
        else:
            return self.fn(data[self.input_col])  # type: ignore[misc]

    def __iter__(self) -> Iterator[T_co]:
        if not self._enabled:  # equivalent to flatmap
            for x in self.datapipe:
                yield from self._apply_fn(x)
        else:
            idx = self._rng.randint(0, self.buffer_size - 1)
            for x in self.datapipe:
                while len(self._buffer) == self.buffer_size:
                    try:
                        yield next(self._buffer[idx])
                        idx = self._rng.randint(0, self.buffer_size - 1)
                    except StopIteration:
                        self._buffer.pop(idx)
                self._buffer.append(iter(self._apply_fn(x)))
            while self._buffer:
                try:
                    idx = self._rng.randint(0, len(self._buffer) - 1)
                    yield next(self._buffer[idx])
                except StopIteration:
                    self._buffer.pop(idx)

    def __len__(self) -> int:
        if self._no_op_fn:
            return sum(map(len, self.datapipe))
        raise TypeError(f"{type(self).__name__}'s length relies on the output of its function.")

    def __getstate__(self):
        state = (
            self.datapipe,
            self.fn,
            self.input_col,
            self.buffer_size,
            self._buffer,
            self._enabled,
            self._seed,
            self._rng.getstate(),
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.fn,
            self.input_col,
            self.buffer_size,
            self._buffer,
            self._enabled,
            self._seed,
            rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)

    def __del__(self):
        self._buffer.clear()


@functional_datapipe("drop")
class DropperIterDataPipe(IterDataPipe[T_co]):
    r"""
    Drop columns/elements in input DataPipe via its indices (functional name: ``drop``).

    Args:
        datapipe: IterDataPipe with columns to be dropped
        indices: a single column index to be dropped or a list of indices

            - Integer(s) is/are used for list/tuple.
            - Key(s) is/are used for dict.

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


@functional_datapipe("slice")
class SliceIterDataPipe(IterDataPipe[T_co]):
    r"""
    returns a slice of elements in input DataPipe via start/stop/step or indices (functional name: ``slice``).

    Args:
        datapipe: IterDataPipe with iterable elements
        index: a single start index for the slice or a list of indices to be returned instead of a start/stop slice

            - Integer(s) is/are used for list/tuple.
            - Key(s) is/are used for dict.


        stop: the slice stop. ignored if index is a list or if element is a dict
        step: step to be taken from start to stop. ignored if index is a list or if element is a dict

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper([(0, 10, 100), (1, 11, 111), (2, 12, 122), (3, 13, 133), (4, 14, 144)])
        >>> slice_dp = dp.slice(0, 2)
        >>> list(slice_dp)
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

        if isinstance(index, list):
            if stop or step:
                warnings.warn(
                    "A list of indices was passed as well as a stop or step for the slice, "
                    "these arguments can't be used together so only the indices list will be used."
                )

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
                elif self.index in old_item.keys():
                    new_item = {self.index: old_item.get(self.index)}  # type: ignore[assignment]
                else:
                    new_item = old_item  # type: ignore[assignment]
                    warnings.warn(
                        "Dictionaries are not sliced by steps, only direct index. "
                        "Please be aware that no filter was done or new item created."
                    )
            else:
                new_item = old_item  # type: ignore[assignment]
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


@functional_datapipe("flatten")
class FlattenIterDataPipe(IterDataPipe[T_co]):
    r"""
    returns a flattened copy of the input DataPipe at the per sample/element level based on provided indices (functional name: ``flatten``).

    Note:
        no args will flatten each item in the datapipe 1 level

    Args:
        datapipe: IterDataPipe with iterable elements
        indices: a single index/key for the item to flatten from an iterator item or a list of indices/keys to be flattened

            - Integer(s) is/are used for list/tuple.
            - Key(s) is/are used for dict.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper([(0, 10, (100, 1000)), (1, 11, (111, 1001)), (2, 12, (122, 1002)), (3, 13, (133, 1003)), (4, 14, (144, 1004))])
        >>> flatten_dp = dp.flatten(2)
        >>> list(flatten_dp)
        [(0, 10, 100, 1000), (1, 11, 111, 1001), (2, 12, 122, 1002), (3, 13, 133, 1003), (4, 14, 144, 1004)]
        >>>
        >>> dp = IterableWrapper([(0, (1, 2)), (3, (4, 5)), (6, (7, 8))])
        >>> flatten_dp = dp.flatten()
        >>> list(flatten_dp)
        [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    datapipe: IterDataPipe
    indices: Set[Hashable] = set()

    def __init__(
        self,
        datapipe: IterDataPipe,
        indices: Optional[Union[Hashable, List[Hashable]]] = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        if indices:
            if isinstance(indices, list):
                self.indices = set(indices)
            else:
                self.indices = {indices}

    def __iter__(self) -> Iterator[T_co]:
        flatten_all = False
        if not self.indices:
            flatten_all = True
        for old_item in self.datapipe:
            if isinstance(old_item, dict):
                new_item = {}  # type: ignore[assignment]
                for k, v in old_item.items():
                    if k in self.indices:
                        pass
                    if (flatten_all or (k in self.indices)) and isinstance(v, dict):
                        for k_sub, v_sub in v.items():
                            if k_sub not in old_item:
                                new_item[k_sub] = v_sub
                            else:
                                warnings.warn(
                                    "Flattener tried to insert the same key twice into the dict item,"
                                    "the second key,value pair has been dropped."
                                )
                    else:
                        if k not in new_item:
                            new_item[k] = v
                        else:
                            warnings.warn(
                                "Flattener tried to insert the same key twice into the dict item,"
                                "the second key,value pair has been dropped."
                            )
            else:
                is_tuple = False
                new_item = []  # type: ignore[assignment]
                if isinstance(old_item, tuple):
                    is_tuple = True
                    old_item = list(old_item)
                for i, item in enumerate(old_item):
                    if (flatten_all or (i in self.indices)) and isinstance(item, (list, tuple)):
                        new_item.extend(list(item))  # type: ignore[attr-defined]
                    else:
                        new_item.append(item)  # type: ignore[attr-defined]
                if is_tuple:
                    new_item = tuple(new_item)  # type: ignore[assignment]

            # check to make sure all indices requested were in the item. warn if not
            try:
                if self.indices:
                    for index in self.indices:
                        old_item[index]
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


class _BatchAsyncMapperIterDataPipe(IterDataPipe):
    datapipe: IterDataPipe
    async_fn: Callable

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        async_fn: Callable,
        input_col=None,
        output_col=None,
        max_concurrency: int = 32,
    ):
        self.source_datapipe = source_datapipe
        if not inspect.iscoroutinefunction(async_fn):
            raise ValueError(f"Expected a corotine function with an async def syntax, but got a {type(async_fn)}")
        self.async_fn = async_fn  # type: ignore[assignment]
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        self.input_col = input_col
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        self.output_col = output_col
        self.max_concurrency = max_concurrency

    def __iter__(self):
        policy = asyncio.get_event_loop_policy()
        loop = policy.new_event_loop()
        try:
            for batch in self.source_datapipe:
                policy.set_event_loop(loop)
                new_batch = loop.run_until_complete(self.processbatch(batch))
                yield new_batch
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def processbatch(self, batch):
        sem = asyncio.Semaphore(self.max_concurrency)

        async def controlled_async_fn(async_fn, *data):
            async with sem:
                return await async_fn(*data)

        coroutines = []
        if self.input_col is None:
            for data in batch:
                coroutines.append(controlled_async_fn(self.async_fn, data))
            results = await asyncio.gather(*coroutines)
            return results

        for data in batch:
            if isinstance(self.input_col, (list, tuple)):
                args = tuple(data[col] for col in self.input_col)
                coroutines.append(controlled_async_fn(self.async_fn, *args))
            else:
                coroutines.append(controlled_async_fn(self.async_fn, data[self.input_col]))
        results = await asyncio.gather(*coroutines)

        new_batch = []
        for data, res in zip(batch, results):
            t_flag = isinstance(data, tuple)
            if t_flag:
                data = list(data)

            if self.output_col is None:
                if isinstance(self.input_col, (list, tuple)):
                    data[self.input_col[0]] = res
                    for idx in sorted(self.input_col[1:], reverse=True):
                        del data[idx]
                else:
                    data[self.input_col] = res
            elif self.output_col == -1:
                data.append(res)
            else:
                data[self.output_col] = res

            if t_flag:
                data = tuple(data)

            new_batch.append(data)
        return new_batch

    def __len__(self):
        return len(self.source_datapipe)


@functional_datapipe("async_map_batches")
class BatchAsyncMapperIterDataPipe(IterDataPipe):
    r"""
    Combines elements from the source DataPipe to batches and applies a coroutine function
    over each element within the batch concurrently, then flattens the outpus to a
    single, unnested IterDataPipe (functional name: ``async_map_batches``).

    Args:
        source_datapipe: Source IterDataPipe
        async_fn: The coroutine function to be applied to each batch of data
        batch_size: The size of batch to be aggregated from ``source_datapipe``
        input_col: Index or indices of data which ``fn`` is applied, such as:

            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

        output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
            only when ``input_col`` is not ``None``

            - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
              multiple indices, the left-most one is used, and other indices will be removed.
            - Integer is used for list/tuple. ``-1`` represents to append result at the end.
            - Key is used for dict. New key is acceptable.

        max_concurrency: Maximum concurrency to call async functions. (Default: ``32``)
        flatten: Determine if the batches get flatten in the end (Default: ``True``)
                 If ``False``, outputs will be in batches of size ``batch_size``

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> async def mul_ten(x):
        ...     await asyncio.sleep(1)
        ...     return x * 10
        >>> dp = IterableWrapper(range(50))
        >>> dp = dp.async_map_batches(mul_ten, 16)
        >>> list(dp)
        [0, 10, 20, 30, ...]
        >>> dp = IterableWrapper([(i, i) for i in range(50)])
        >>> dp = dp.async_map_batches(mul_ten, 16, input_col=1)
        >>> list(dp)
        [(0, 0), (1, 10), (2, 20), (3, 30), ...]
        >>> dp = IterableWrapper([(i, i) for i in range(50)])
        >>> dp = dp.async_map_batches(mul_ten, 16, input_col=1, output_col=-1)
        >>> list(dp)
        [(0, 0, 0), (1, 1, 10), (2, 2, 20), (3, 3, 30), ...]
        # Async fetching html from remote
        >>> from aiohttp import ClientSession
        >>> async def fetch_html(url: str, **kwargs):
        ...     async with ClientSession() as session:
        ...         resp = await session.request(method="GET", url=url, **kwargs)
        ...         resp.raise_for_status()
        ...         html = await resp.text()
        ...     return html
        >>> dp = IterableWrapper(urls)
        >>> dp = dp.async_map_batches(fetch_html, 16)
    """

    def __new__(
        self,
        source_datapipe,
        async_fn: Callable,
        batch_size: int,
        input_col=None,
        output_col=None,
        max_concurrency: int = 32,
        flatten: bool = True,
    ):
        dp = source_datapipe.batch(batch_size)
        dp = _BatchAsyncMapperIterDataPipe(dp, async_fn, input_col, output_col, max_concurrency)
        if flatten:
            dp = dp.flatmap()
            try:
                source_length = len(source_datapipe)
                if isinstance(source_length, int) and source_length >= 0:
                    dp = dp.set_length(source_length)
            except (TypeError, NotImplementedError):
                pass
        return dp


@functional_datapipe("threadpool_map")
class ThreadPoolMapperIterDataPipe(IterDataPipe[T_co]):
    r"""
    Applies a function over each item from the source DataPipe concurrently
    using ``ThreadPoolExecutor`` (functional name: ``threadpool_map``).
    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        source_datapipe: Source IterDataPipe
        fn: Function being applied over each item
        input_col: Index or indices of data which ``fn`` is applied, such as:

            - ``None`` as default to apply ``fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

        output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
            only when ``input_col`` is not ``None``

            - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
              multiple indices, the left-most one is used, and other indices will be removed.
            - Integer is used for list/tuple. ``-1`` represents to append result at the end.
            - Key is used for dict. New key is acceptable.

        scheduled_tasks: How many tasks will be scheduled at any given time (Default value: 128)
        max_workers: Maximum number of threads to execute function calls
        **threadpool_kwargs: additional arguments to be given to the ``ThreadPoolExecutor``

    Note:
         For more information about ``max_workers`` and additional arguments for the ``ThreadPoolExecutor``
         please refer to: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor

    Note:
        For optimal use of all threads, ``scheduled_tasks`` > ``max_workers`` is strongly recommended. The higher the
        variance of the time needed to finish execution of the given ``fn`` is, the higher the value
        of ``scheduled_tasks`` needs to be to avoid threads sitting idle while waiting
        for the next result (as results are returned in correct order).

        However, too high value of ``scheduled_tasks`` might lead to long waiting period until the first element is yielded
        as ``next`` is called ``scheduled_tasks`` many times on ``source_datapipe`` before yielding.

        We encourage you to try out different values of ``max_workers`` and ``scheduled_tasks``
        in search for optimal values for your use-case.

    Example:

    .. testsetup::

        from torchdata.datapipes.iter import IterableWrapper
        import requests
        import time
        from unittest.mock import MagicMock

        requests.get = MagicMock()
        urls = []

    .. testcode::

        # fetching html from remote
        def fetch_html(url: str, **kwargs):
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r.content
        dp = IterableWrapper(urls)
        dp = dp.threadpool_map(fetch_html,max_workers=16)

    .. testcode::

        def mul_ten(x):
            time.sleep(0.1)
            return x * 10

        dp = IterableWrapper([(i, i) for i in range(50)])
        dp = dp.threadpool_map(mul_ten, input_col=1)
        print(list(dp))

    .. testoutput::

        [(0, 0), (1, 10), (2, 20), (3, 30), ...]

    .. testcode::

        dp = IterableWrapper([(i, i) for i in range(50)])
        dp = dp.threadpool_map(mul_ten, input_col=1, output_col=-1)
        print(list(dp))

    .. testoutput::

        [(0, 0, 0), (1, 1, 10), (2, 2, 20), (3, 3, 30), ...]

    """

    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        fn: Callable,
        input_col=None,
        output_col=None,
        scheduled_tasks: int = 128,
        max_workers: Optional[int] = None,
        **threadpool_kwargs,
    ) -> None:
        super().__init__()
        self.datapipe = source_datapipe

        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

        if scheduled_tasks <= 0:
            raise ValueError("'scheduled_tasks' is required to be a positive integer.")
        self.scheduled_tasks = scheduled_tasks
        if max_workers is not None and max_workers <= 0:
            raise ValueError("'max_workers' is required to be a positive integer.")
        self.max_workers = max_workers
        self.threadpool_kwargs = threadpool_kwargs

        self.input_col = input_col
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        self.output_col = output_col
        validate_input_col(fn, input_col)

    def _apply_fn(self, data):
        if self.input_col is None and self.output_col is None:
            return self.fn(data)

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

        if self.output_col is None:
            if isinstance(self.input_col, (list, tuple)):
                data[self.input_col[0]] = res
                for idx in sorted(self.input_col[1:], reverse=True):
                    del data[idx]
            else:
                data[self.input_col] = res
        else:
            if self.output_col == -1:
                data.append(res)
            else:
                data[self.output_col] = res

        # Convert list back to tuple
        return tuple(data) if t_flag else data

    def __iter__(self) -> Iterator[T_co]:
        with futures.ThreadPoolExecutor(max_workers=self.max_workers, **self.threadpool_kwargs) as executor:
            futures_deque: deque = deque()
            has_next = True
            itr = iter(self.datapipe)
            for _ in range(self.scheduled_tasks):
                try:
                    futures_deque.append(executor.submit(self._apply_fn, next(itr)))
                except StopIteration:
                    has_next = False
                    break

            while len(futures_deque) > 0:
                if has_next:
                    try:
                        futures_deque.append(executor.submit(self._apply_fn, next(itr)))
                    except StopIteration:
                        has_next = False
                yield futures_deque.popleft().result()

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
