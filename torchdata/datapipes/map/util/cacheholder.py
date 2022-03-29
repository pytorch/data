# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, Optional, Set, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.map import MapDataPipe


T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("in_memory_cache")
class InMemoryCacheHolderMapDataPipe(MapDataPipe[T_co]):
    r"""
    Stores elements from the source DataPipe in memory (functional name: ``in_memory_cache``).

    The default ``cache`` is a ``Dict``. Depending on the use case and shape of the data, it may be useful to use other
    objects as the cache, such as an LRU cache, ``numpy.array``, or ``multiprocessing.Array``.
    to cache data in shared memory.

    Args:
        source_dp: source DataPipe from which elements are read and stored in memory
        cache: a cache object that implements ``__getitem__`` and ``__setitem__``

    Example:
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> source_dp = SequenceWrapper(range(10))
        >>> cache_dp = source_dp.in_memory_cache()
        >>> list(cache_dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, source_dp: MapDataPipe[T_co], cache: Optional[Callable] = None) -> None:
        self.source_dp: MapDataPipe[T_co] = source_dp
        self.cache = {} if cache is None else cache
        if not (hasattr(self.cache, "__getitem__") and hasattr(self.cache, "__setitem__")):
            raise TypeError("`cache` must be an object with  both '__getitem__' and '__setitem__.")
        self.index_cached: Set = set()

    def __getitem__(self, index) -> T_co:
        if index not in self.index_cached:
            self.cache[index] = self.source_dp[index]  # type: ignore[index]
            self.index_cached.add(index)
        return self.cache[index]  # type: ignore[index]

    def __len__(self) -> int:
        return len(self.source_dp)
