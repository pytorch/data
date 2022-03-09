# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Iterator, TypeVar
from warnings import warn

import torch.distributed as dist

from torch.utils.data import get_worker_info
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("header")
class HeaderIterDataPipe(IterDataPipe[T_co]):
    r"""
    Yields elements from the source DataPipe from the start, up to the specfied limit (functional name: ``header``).

    Args:
        source_datapipe: the DataPipe from which elements will be yielded
        limit: the number of elements to yield before stopping

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> header_dp = dp.header(3)
        >>> list(header_dp)
        [0, 1, 2]
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], limit: int = 10) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.limit: int = limit
        self.length: int = -1

    def __iter__(self) -> Iterator[T_co]:
        i: int = 0
        for value in self.source_datapipe:
            i += 1
            if i <= self.limit:
                yield value
            else:
                break
        self.length = min(i, self.limit)  # We know length with certainty when we reach here

    def __len__(self) -> int:
        if self.length != -1:
            return self.length
        try:
            source_len = len(self.source_datapipe)
            self.length = min(source_len, self.limit)
            return self.length
        except TypeError:
            warn(
                "The length of this HeaderIterDataPipe is inferred to be equal to its limit."
                "The actual value may be smaller if the actual length of source_datapipe is smaller than the limit."
            )
            return self.limit


@functional_datapipe("limit")
class LimiterIterDataPipe(IterDataPipe[T_co]):
    """
    Limits the number of samples that the source DataPipe can yield based the input as well as the number of
    processes and workers if distribute- or multi-processing is enabled (functional name: ``limit``).

    The number of samples that will be yielded equals ``num_take // (num_processes * worker_info.num_workers)``.

    Args:
        source_datapipe: IterDataPipe with samples of data that will be yielded
        num_take: the number of samples that you wish to yield across all processes and workers, should be less than
            or equal to the ``len(source_datapipe)``

    Example:
        >>> source_dp = IterableWrapper(range(20))
        >>> limiter_dp = source_dp.limit(5)
        >>> list(limiter_dp)
        [0, 1, 2, 3, 4]
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], num_take: int) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.num_take: int = num_take
        self.num_processes: int = 1
        if dist.is_available() and dist.is_initialized():
            self.num_processes = dist.get_world_size()

    def _get_num_workers(self):
        num_workers = self.num_processes
        worker_info = get_worker_info()
        if worker_info is not None:
            # Note that we are assuming each distributed process has the same number of workers
            num_workers *= worker_info.num_workers
        return num_workers

    def __iter__(self) -> Iterator:
        # TODO: pmeier - this is weird as it drops more elements than it should
        worker_num_take = self.num_take // self._get_num_workers()

        for i, data in enumerate(self.source_datapipe):
            # TODO: If each instance of the source_datapipe (of different worker/process) is not independently
            #  shuffled, then all Limiter will return the same set of samples
            #  Should we return samples based on process_id and worker_id?
            #  Alternatively, should we optionally allow people to shuffle inside `__init__`?
            if i < worker_num_take:
                yield data
            else:
                break

    # TODO: What is the length that user wants here
    #       1) The total dataset length, or 2) the length that the specific worker/process will yield
    def __len__(self) -> int:
        max_num_take = self.num_take
        try:
            if len(self.source_datapipe) < self.num_take:
                max_num_take = len(self.source_datapipe)
        except TypeError:
            pass
        return max_num_take // self._get_num_workers()
