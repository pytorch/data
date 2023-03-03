# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, Optional, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("cycle")
class CyclerIterDataPipe(IterDataPipe[T_co]):
    """
    Cycles the specified input in perpetuity by default, or for the specified number
    of times (functional name: ``cycle``).

    If the ordering does not matter (e.g. because you plan to ``shuffle`` later) or if you would like to
    repeat an element multiple times before moving onto the next element, use :class:`.Repeater`.

    Args:
        source_datapipe: source DataPipe that will be cycled through
        count: the number of times to read through ``source_datapipe` (if ``None``, it will cycle in perpetuity)

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(3))
        >>> dp = dp.cycle(2)
        >>> list(dp)
        [0, 1, 2, 0, 1, 2]
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


@functional_datapipe("repeat")
class RepeaterIterDataPipe(IterDataPipe[T_co]):
    """
    Repeatedly yield each element of source DataPipe for the specified number of times before
    moving onto the next element (functional name: ``repeat``). Note that no copy is made in this DataPipe,
    the same element is yielded repeatedly.

    If you would like to yield the whole DataPipe in order multiple times, use :class:`.Cycler`.

    Args:
        source_datapipe: source DataPipe that will be iterated through
        times: the number of times an element of ``source_datapipe`` will be yielded before moving onto the next element

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(3))
        >>> dp = dp.repeat(2)
        >>> list(dp)
        [0, 0, 1, 1, 2, 2]
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], times: int) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.times: int = times
        if times <= 1:
            raise ValueError(f"The number of repetition must be > 1, got {times}")

    def __iter__(self) -> Iterator[T_co]:
        for element in self.source_datapipe:
            for _ in range(self.times):
                yield element

    def __len__(self) -> int:
        return self.times * len(self.source_datapipe)
