# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterator, List, Optional, Set, Sized, Tuple

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


@functional_datapipe("zip_longest")
class ZipperLongestIterDataPipe(IterDataPipe):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip_longest``).
    The output is stopped until all input DataPipes are exhausted. If any input DataPipe is exhausted,
    missing values are filled-in with `fill_value` (default value is None).

    Args:
        *datapipes: Iterable DataPipes being aggregated
        *fill_value: Value that user input to fill in the missing values from DataPipe. Default value is None.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.zip_longest(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (None, 13, 23), (None, 14, 24)]
        >>> list(dp1.zip_longest(dp2, dp3, -1))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (-1, 13, 23), (-1, 14, 24)]
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]
    fill_value: Any

    def __init__(
        self,
        *datapipes: IterDataPipe,
        fill_value: Any = None,
    ):
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs are required to be `IterDataPipe` " "for `ZipperLongestIterDataPipe`.")
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]
        self.fill_value = fill_value

    def __iter__(self) -> Iterator[Tuple]:
        iterators = [iter(x) for x in self.datapipes]
        finished: Set[int] = set()
        while len(finished) < len(iterators):
            values: List[Any] = []
            for i in range(len(iterators)):
                value = self.fill_value
                if i not in finished:
                    try:
                        value = next(iterators[i])
                    except StopIteration:
                        finished.add(i)
                        if len(finished) == len(iterators):
                            return
                values.append(value)
            yield tuple(values)

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return max(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
