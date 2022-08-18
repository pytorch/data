# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, TypeVar
from warnings import warn

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("header")
class HeaderIterDataPipe(IterDataPipe[T_co]):
    r"""
    Yields elements from the source DataPipe from the start, up to the specfied limit (functional name: ``header``).

    If you would like to manually set the length of a DataPipe to a certain value; we recommend you to
    use :class:`.LengthSetter`.

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


@functional_datapipe("set_length")
class LengthSetterIterDataPipe(IterDataPipe[T_co]):
    r"""
    Set the length attribute of the DataPipe, which is returned by ``__len__`` (functional name: ``set_length``).
    This can be used after DataPipes whose final length cannot be known in advance (e.g. ``filter``). If you
    know the final length with certainty, you can manually set it for usages by DataLoader or other DataPipes.

    This DataPipe differs from :class:`.Header` in that this doesn't restrict the number of elements that
    can be yielded from the DataPipe; this is strictly used for setting an attribute so that it can be used later.

    Args:
        source_datapipe: a DataPipe
        length: the integer value that will be set as the length

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10)).filter(lambda x: x < 5).set_length(5)
        >>> list(dp)
        [0, 1, 2, 3, 4]
        >>> len(dp)
        5
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], length: int) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[T_co]:
        yield from self.source_datapipe

    def __len__(self) -> int:
        return self.length
