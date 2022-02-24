# Copyright (c) Facebook, Inc. and its affiliates.
import random
from typing import Callable, Optional, TypeVar

from torch.utils.data import DataChunk

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


def _in_batch_shuffle_fn(data: DataChunk):
    d = list(data)
    random.shuffle(d)
    return DataChunk(d)


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
        in_batch_shuffle: iF True, do in-batch shuffle; if False, buffer shuffle

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
        >>>     bucket_num=1, in_batch_shuffle=False, sort_key=sort_bucket
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
    in_batch_shuffle: bool
    length: Optional[int]

    def __new__(
        cls,
        datapipe: IterDataPipe[T_co],
        batch_size: int,
        drop_last: bool = False,
        batch_num: int = 100,
        bucket_num: int = 1,
        sort_key: Optional[Callable] = None,
        in_batch_shuffle: bool = True,
    ):
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        assert batch_num > 0, "Number of batches is required to be larger than 0!"
        assert bucket_num > 0, "Number of buckets is required to be larger than 0!"

        bucket_size = batch_size * batch_num
        pool_size = bucket_size * bucket_num

        # Shuffle by pool_size
        if bucket_num > 1 or sort_key is None:
            if in_batch_shuffle:
                datapipe = datapipe.batch(batch_size=pool_size, drop_last=False).map(fn=_in_batch_shuffle_fn).unbatch()
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
            if in_batch_shuffle:
                datapipe = datapipe.batch(batch_size=bucket_num, drop_last=False).map(fn=_in_batch_shuffle_fn).unbatch()
            else:
                datapipe = datapipe.shuffle(buffer_size=bucket_size)
        return datapipe
