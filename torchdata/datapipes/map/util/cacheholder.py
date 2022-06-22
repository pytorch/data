# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.map import MapDataPipe


T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("in_memory_cache")
class InMemoryCacheHolderMapDataPipe(MapDataPipe[T_co]):
    r"""
    Stores elements from the source DataPipe in memory (functional name: ``in_memory_cache``). Once an item is
    stored, it will remain unchanged and subsequent retrivals will return the same element. Since items from
    ``MapDataPipe`` are lazily computed, this can be used to store the results from previous ``MapDataPipe`` and
    reduce the number of duplicate computations.

    Note:
        The default ``cache`` is a ``Dict``. If another data structure is more suitable as cache for your use

    Args:
        source_dp: source DataPipe from which elements are read and stored in memory

    Example:
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> source_dp = SequenceWrapper(range(10))
        >>> cache_dp = source_dp.in_memory_cache()
        >>> list(cache_dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, source_dp: MapDataPipe[T_co]) -> None:
        self.source_dp: MapDataPipe[T_co] = source_dp
        self.cache: Dict[Any, T_co] = {}

    def __getitem__(self, index) -> T_co:
        if index not in self.cache:
            self.cache[index] = self.source_dp[index]  # type: ignore[index]
        return self.cache[index]  # type: ignore[index]
        # We can potentially remove `self.source_dp` to save memory once `len(self.cache) == len(self.source_dp)`
        # Be careful about how that may interact with and graph traversal and other features

    def __len__(self) -> int:
        return len(self.source_dp)
