# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Iterator, Optional, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("cycle")
class CyclerIterDataPipe(IterDataPipe[T_co]):
    """
    Cycle the specified input in perpetuity (by default), or for the specified number of times.

    Args:
        source_datapipe: source DataPipe that will be cycled through
        count: the number of times to read through the source DataPipe (if `None`, it will cycle in perpetuity)
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], count: Optional[int] = None) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.count: Optional[int] = count
        if count is not None and count < 0:
            raise ValueError(f"Expected non-negative count, got {count}")

    def __iter__(self) -> Iterator[T_co]:
        i = 0
        while self.count is None or i < self.count:
            yield from self.source_datapipe
            i += 1

    def __len__(self) -> int:
        if self.count is None:
            raise TypeError(
                f"This {type(self).__name__} instance cycles forever, and therefore doesn't have valid length"
            )
        else:
            return self.count * len(self.source_datapipe)
