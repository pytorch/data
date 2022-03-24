# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, List, Optional, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.map import MapDataPipe


T_co = TypeVar("T_co", covariant=True)


def list_cache(size: int) -> List:
    cache = [None] * size
    try:
        from numpy import array

        cache = array(cache)
    except (ImportError, ModuleNotFoundError):
        pass
    return cache


@functional_datapipe("in_memory_cache")
class InMemoryCacheHolderMapDataPipe(MapDataPipe[T_co]):
    r"""
    Stores elements from the source DataPipe in memory (functional name: ``in_memory_cache``).

    Args:
        source_dp: source DataPipe from which elements are read and stored in memory
        cache_fn: function that takes in an integer size and return a cache object that implements ``__getitem__``

    Example:
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> source_dp = SequenceWrapper(range(10))
        >>> cache_dp = source_dp.in_memory_cache()
        >>> list(cache_dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, source_dp: MapDataPipe[T_co], cache_fn: Optional[Callable] = None) -> None:
        self.source_dp: MapDataPipe[T_co] = source_dp
        if cache_fn is None:
            cache_fn = list_cache
        self.cache = cache_fn(len(source_dp))
        if not (hasattr(self.cache, "__getitem__") and hasattr(self.cache, "__setitem__")):
            raise TypeError("The output of `cache_fn` must be an object with  both '__getitem__' and '__setitem__.")
        self.index_cached = set()

    def __getitem__(self, index) -> T_co:
        if index not in self.index_cached:
            self.cache[index] = self.source_dp[index]
        return self.cache[index]

    def __len__(self) -> int:
        return len(self.source_dp)
