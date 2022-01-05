# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Iterator, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("header")
class HeaderIterDataPipe(IterDataPipe[T_co]):
    r"""
    Iterable DataPipe that yields elements from the source DataPipe from the start up to the given limit

    Args:
        source_datapipe: the DataPipe from which elements will be yielded
        limit: the number of elements to yield before stopping
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], limit: int = 10) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.limit: int = limit

    def __iter__(self) -> Iterator[T_co]:
        for i, value in enumerate(self.source_datapipe):
            if i < self.limit:
                yield value
            else:
                break

    # TODO(134): Fix the case that the length of source_datapipe is shorter than limit
    def __len__(self) -> int:
        return self.limit
