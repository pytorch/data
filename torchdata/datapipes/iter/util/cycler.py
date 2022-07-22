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

    def __init__(
        self, source_datapipe: IterDataPipe[T_co], count: Optional[int] = None
    ) -> None:
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
