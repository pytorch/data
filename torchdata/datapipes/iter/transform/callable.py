# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import inspect
import warnings
from concurrent import futures

from typing import Callable, Hashable, Iterator, List, Optional, Set, Sized, TypeVar, Union

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


def _merge_batch_with_result(orig_batch, results, input_col, output_col):
    if input_col is None:
        return results

    new_batch = []
    for data, res in zip(orig_batch, results):
        t_flag = isinstance(data, tuple)
        if t_flag:
            data = list(data)

        if output_col is None:
            if isinstance(input_col, (list, tuple)):
                data[input_col[0]] = res
                for idx in sorted(input_col[1:], reverse=True):
                    del data[idx]
            else:
                data[input_col] = res
        elif output_col == -1:
            data.append(res)
        else:
            data[output_col] = res

        if t_flag:
            data = tuple(data)

        new_batch.append(data)
    return new_batch


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
        return _merge_batch_with_result(batch, results, self.input_col, self.output_col)


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
        max_concurrency: Maximum concurrency to call async functions. (Default value: 32)

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
    ):
        dp = source_datapipe.batch(batch_size)
        dp = _BatchAsyncMapperIterDataPipe(dp, async_fn, input_col, output_col, max_concurrency)
        dp = dp.flatmap()
        return dp


class _BatchThreadPoolMapperIterDataPipe(IterDataPipe):
    datapipe: IterDataPipe
    fn: Callable

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        fn: Callable,
        input_col=None,
        output_col=None,
        max_workers: Optional[int] = None,
        **threadpool_kwargs,
    ):
        self.source_datapipe = source_datapipe
        self.fn = fn
        self.input_col = input_col
        validate_input_col(fn, input_col)
        if input_col is None and output_col is not None:
            raise ValueError("`output_col` must be None when `input_col` is None.")
        if isinstance(output_col, (list, tuple)):
            if len(output_col) > 1:
                raise ValueError("`output_col` must be a single-element list or tuple")
            output_col = output_col[0]
        if isinstance(self.input_col, (list, tuple)):

            def wrapper_fn(args):
                return fn(*args)

            self.fn = wrapper_fn
        self.output_col = output_col
        self.max_workers = max_workers
        self.threadpool_kwargs = threadpool_kwargs

    def __iter__(self):
        with futures.ThreadPoolExecutor(max_workers=self.max_workers, **self.threadpool_kwargs) as executor:
            for batch in self.source_datapipe:
                prepared_batch = self.preparebatch(batch)
                results = executor.map(self.fn, prepared_batch)
                yield _merge_batch_with_result(batch, results, self.input_col, self.output_col)

    def preparebatch(self, batch):
        if self.input_col is None:
            return batch

        prepared_batch = []
        for data in batch:
            if isinstance(self.input_col, (list, tuple)):
                args = tuple(data[col] for col in self.input_col)
                prepared_batch.append(args)
            else:
                prepared_batch.append(data[self.input_col])
        return prepared_batch


@functional_datapipe("thread_map_batches")
class BatchThreadPoolMapperIterDataPipe(IterDataPipe):
    r"""
    Combines elements from the source DataPipe to batches and applies a function
    over each element within the batch concurrently using ``ThreadPoolExecutor``, then flattens the output to a
    single, unnested IterDataPipe (functional name: ``thread_map_batches``).

    Args:
        source_datapipe: Source IterDataPipe
        fn: The function to be applied to each element within batch of data
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
        max_workers: Maximum num of threads to execute function calls. (Default value: None)

    Note:
         For more information about ``max_workers`` and please refer to: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor

    Example:

    .. testsetup::

        from torchdata.datapipes.iter import IterableWrapper
        import requests
        import time
        from unittest.mock import MagicMock

        requests.get = MagicMock()
        urls = []

    .. testcode::

        def mul_ten(x):
            time.sleep(0.1)
            return x * 10
        dp = IterableWrapper(range(50))
        dp = dp.thread_map_batches(mul_ten, 16)
        print(list(dp))

    .. testoutput::

        [0, 10, 20, 30, ...]

    .. testcode::

        dp = IterableWrapper([(i, i) for i in range(50)])
        dp = dp.thread_map_batches(mul_ten, 16, input_col=1)
        print(list(dp))

    .. testoutput::

        [(0, 0), (1, 10), (2, 20), (3, 30), ...]

    .. testcode::

        dp = IterableWrapper([(i, i) for i in range(50)])
        dp = dp.thread_map_batches(mul_ten, 16, input_col=1, output_col=-1)
        print(list(dp))

    .. testoutput::

        [(0, 0, 0), (1, 1, 10), (2, 2, 20), (3, 3, 30), ...]

    .. testcode::

        # fetching html from remote
        def fetch_html(url: str, **kwargs):
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r.content
        dp = IterableWrapper(urls)
        dp = dp.thread_map_batches(fetch_html, 16)

    """

    def __new__(
        cls,
        source_datapipe,
        fn: Callable,
        batch_size: int,
        input_col=None,
        output_col=None,
        max_workers: Optional[int] = None,
        **threadpool_kwargs,
    ):
        dp = source_datapipe.batch(batch_size)
        dp = _BatchThreadPoolMapperIterDataPipe(dp, fn, input_col, output_col, max_workers, **threadpool_kwargs)
        dp = dp.flatmap()
        return dp
