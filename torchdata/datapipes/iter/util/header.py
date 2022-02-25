# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Iterator, TypeVar
from warnings import warn

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
