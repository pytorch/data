# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import heapq
import random

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, final, Generic, Iterator, List, Optional, TypeVar

import torch

from torchdata.datapipes import DataChunk, functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("in_batch_shuffle")
class InBatchShufflerIterDataPipe(IterDataPipe[DataChunk[T_co]]):
    r"""
    Shuffles each mini-batch from the prior DataPipe (functional name: ``in_batch_shuffle``).

    Args:
        datapipe: Iterable DataPipe with batched data

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> batch_dp = source_dp.batch(batch_size=3, drop_last=True)
        >>> list(batch_dp)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> in_batch_shuffle_dp = batch_dp.in_batch_shuffle()
        >>> list(in_batch_shuffle_dp)
        [[2, 0, 1], [3, 5, 4], [7, 8, 6]]
    """

    def __init__(self, datapipe: IterDataPipe[DataChunk[T_co]]):
        self.datapipe = datapipe
        self._enabled = True
        self._seed: Optional[int] = None
        self._rng = random.Random()

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[DataChunk[T_co]]:
        if not self._enabled:
            for batch in self.datapipe:
                yield batch
        else:
            for batch in self.datapipe:
                new_batch = self._rng.sample(batch, len(batch))
                yield DataChunk(new_batch)

    @final
    def reset(self) -> None:
        if self._enabled:
            if self._seed is None:
                self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self._rng.seed(self._seed)
            self._seed = None

    def __len__(self) -> int:
        return len(self.datapipe)

    def __getstate__(self):
        state = (
            self.datapipe,
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
            self._enabled,
            self._seed,
            rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)


@functional_datapipe("bucketbatch")
class BucketBatcherIterDataPipe(IterDataPipe[DataChunk[T_co]]):
    r"""
    Creates mini-batches of data from sorted bucket (functional name: ``bucketbatch``). An outer
    dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
    or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.

    The purpose of this DataPipe is to batch samples with some similarity according to the sorting function
    being passed. For an example in the text domain, it may be batching examples with similar number of tokens
    to minimize padding and to increase throughput.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
        batch_num: Number of batches within a bucket (i.e. `bucket_size = batch_size * batch_num`)
        bucket_num: Number of buckets to consist a pool for shuffling (i.e. `pool_size = bucket_size * bucket_num`)
        sort_key: Callable to sort a bucket (list)
        use_in_batch_shuffle: if True, do in-batch shuffle; if False, buffer shuffle

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> batch_dp = source_dp.bucketbatch(batch_size=3, drop_last=True)
        >>> list(batch_dp)
        [[5, 6, 7], [9, 0, 1], [4, 3, 2]]
        >>> def sort_bucket(bucket):
        >>>     return sorted(bucket)
        >>> batch_dp = source_dp.bucketbatch(
        >>>     batch_size=3, drop_last=True, batch_num=100,
        >>>     bucket_num=1, use_in_batch_shuffle=False, sort_key=sort_bucket
        >>> )
        >>> list(batch_dp)
        [[3, 4, 5], [6, 7, 8], [0, 1, 2]]
    """
    datapipe: IterDataPipe[T_co]
    batch_size: int
    drop_last: bool
    batch_num: int
    bucket_num: int
    sort_key: Optional[Callable]
    use_in_batch_shuffle: bool

    def __new__(
        cls,
        datapipe: IterDataPipe[T_co],
        batch_size: int,
        drop_last: bool = False,
        batch_num: int = 100,
        bucket_num: int = 1,
        sort_key: Optional[Callable] = None,
        use_in_batch_shuffle: bool = True,
    ):
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        assert batch_num > 0, "Number of batches is required to be larger than 0!"
        assert bucket_num > 0, "Number of buckets is required to be larger than 0!"

        bucket_size = batch_size * batch_num
        pool_size = bucket_size * bucket_num

        # Shuffle by pool_size
        if bucket_num > 1 or sort_key is None:
            if use_in_batch_shuffle:
                datapipe = datapipe.batch(batch_size=pool_size, drop_last=False).in_batch_shuffle().unbatch()
            else:
                datapipe = datapipe.shuffle(buffer_size=pool_size)
        # Sort by bucket_size if sort_key is given
        if sort_key is not None:
            datapipe = datapipe.batch(bucket_size).map(fn=sort_key).unbatch()
        # Batch and drop last (if needed)
        datapipe = datapipe.batch(batch_size, drop_last=drop_last)
        # Shuffle the batched data
        if sort_key is not None:
            # In-batch shuffle each bucket seems not that useful, it seems misleading since .batch is called prior.
            if use_in_batch_shuffle:
                datapipe = datapipe.batch(batch_size=bucket_num, drop_last=False).in_batch_shuffle().unbatch()
            else:
                datapipe = datapipe.shuffle(buffer_size=bucket_size)
        return datapipe


def _default_len_fn(token):
    return len(token)


@dataclass(order=True, frozen=True)
class PrioritizedItem(Generic[T_co]):
    length: int
    data: T_co = field(compare=False)


def _token_len_fn(token: T, len_fn: Callable) -> PrioritizedItem[T]:
    return PrioritizedItem(length=len_fn(token), data=token)


def _token_filter_fn(data, *, min_len, max_len):
    return data.length >= min_len and data.length <= max_len


@functional_datapipe("max_token_bucketize")
class MaxTokenBucketizerIterDataPipe(IterDataPipe[DataChunk[T_co]]):
    r"""
    Creates mini-batches of data from a min-heap with limited size, and the total length of samples
    returned by ``len_fn`` within each batch will be limited by ``max_token_count``
    (functional name: ``max_token_bucketize``). If ``min_len`` or ``max_len`` is set, the samples with
    length that is out of ``[min_len, max_len]`` will be filtered out.

    The purpose of this DataPipe is to batch samples with similar length according to ``len_fn``.
    Min-heap is used here to make sure the samples are sorted incrementally based on the length. And,
    the total length of samples in each batch is guaranteed to be smaller than ``max_token_count``.
    For an example in the audio domain, it may be batching samples with similar length. Then, given the
    ``max_token_count``, each batch may be concatenated to a Tensor with the same size and minimum padding.

    If ``include_padding`` is set to ``True``, the token count of each batch includes the padding a succeeding
    DataPipe could add. This guarentees that even after the batch is padded, ``max_token_count`` will not be exceeded.
    This can prevent out-of-memory issues for data with large variations in length.

    Note that batches are bucketized starting from the smallest size in a buffer.
    This can limit the variablity of batches if ``buffer_size`` is large.
    To increase variablity, apply ``torchdata.datapipes.iter.Shuffler`` before and after this DataPipe,
    and keep ``buffer_size`` small.


    Args:
        datapipe: Iterable DataPipe being batched
        max_token_count: Maximum length of total length of data in each batch
        len_fn: Function to be applied to each element to get lengths. ``len(data)`` is used by default.
        min_len: Optional minimum length to be included into each batch
        max_len: Optional maximum length to be included into each batch.
        buffer_size: This restricts how many samples are taken from prior DataPipe to bucketize
        include_padding: If True, the size of each batch includes the extra padding to the largest length in the batch.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(['1', '11', '1', '1111', '111', '1', '11', '11', '111'])
        >>> # Using default len_fn to sort samples based on length (string length in this case)
        >>> batch_dp = source_dp.max_token_bucketize(max_token_count=5)
        >>> list(batch_dp)
        [['1', '1', '1', '11'], ['11', '11'], ['111'], ['111'], ['1111']]
        >>> batch_dp = source_dp.max_token_bucketize(max_token_count=4, buffer_size=4)
        >>> list(batch_dp)
        [['1', '1', '1'], ['11', '11'], ['11'], ['111'], ['111'], ['1111']]
    """
    datapipe: IterDataPipe[PrioritizedItem[T_co]]
    max_token_count: int
    len_fn: Callable
    min_len: int
    max_len: Optional[int]
    buffer_size: int

    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        max_token_count: int,
        len_fn: Callable = _default_len_fn,
        min_len: int = 0,
        max_len: Optional[int] = None,
        buffer_size: int = 1000,
        include_padding: bool = False,
    ) -> None:
        if max_len is None:
            max_len = max_token_count

        if min_len < 0 or min_len > max_len:
            raise ValueError("``min_len`` should be larger than 0 and equal to or smaller than ``max_len``.")
        if max_len > max_token_count:
            raise ValueError("``max_token_count`` must be equal to or greater than ``max_len``.")
        if buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be a positive integer.")
        self.datapipe = datapipe.map(partial(_token_len_fn, len_fn=len_fn))
        self.datapipe = self.datapipe.filter(partial(_token_filter_fn, min_len=min_len, max_len=max_len))
        self.max_token_count = max_token_count
        self.buffer_size = buffer_size
        self.include_padding = include_padding

    def __iter__(self) -> Iterator[DataChunk[T_co]]:
        buffer: List[PrioritizedItem[T_co]] = []
        batch: List[T_co] = []
        batch_size: int = 0
        max_length: int = 0
        for d in self.datapipe:
            heapq.heappush(buffer, d)
            if len(buffer) == self.buffer_size:
                buffer, batch, batch_size, max_length, data_chunk = self._pop_buffer(
                    buffer, batch, batch_size, max_length
                )
                if data_chunk is not None:
                    yield data_chunk
        while buffer:
            buffer, batch, batch_size, max_length, data_chunk = self._pop_buffer(buffer, batch, batch_size, max_length)
            if data_chunk is not None:
                yield data_chunk
        if batch:
            yield DataChunk(batch)

    def _pop_buffer(self, buffer: List[PrioritizedItem[T_co]], batch: List[T_co], batch_size: int, max_length: int):
        data_chunk_to_yield = None
        d: PrioritizedItem[T_co] = heapq.heappop(buffer)
        length = d.length
        token = d.data

        if self.include_padding:
            max_length = max(length, max_length)
            new_batch_size = (len(batch) + 1) * max_length
        else:
            new_batch_size = batch_size + length

        if new_batch_size > self.max_token_count:
            data_chunk_to_yield = DataChunk(batch)
            batch = [token]
            batch_size = length
            max_length = length
        else:
            batch.append(token)
            batch_size = new_batch_size

        return buffer, batch, batch_size, max_length, data_chunk_to_yield
