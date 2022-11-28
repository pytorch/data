# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Sized

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


@functional_datapipe("mux_longest")
class MultiplexerLongestIterDataPipe(IterDataPipe):
    r"""
    Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux_longest``). As in,
    one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
    and so on. It skips over DataPipes that are exhausted, and ends when all input DataPipes are exhausted.

    Args:
        datapipes: Iterable DataPipes that will take turn to yield their elements, until they are all exhausted

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.mux_longest(dp2, dp3))
        [0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24]
    """

    def __init__(self, *datapipes):
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]
        finished: Set[int] = set()
        while len(finished) < len(iterators):
            for i in range(len(iterators)):
                if i not in finished:
                    try:
                        value = next(iterators[i])
                        yield value
                    except StopIteration:
                        finished.add(i)

    def __len__(self):
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return sum(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
