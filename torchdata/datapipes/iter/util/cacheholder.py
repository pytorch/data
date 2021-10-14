# Copyright (c) Facebook, Inc. and its affiliates.
import sys

from collections import deque
from typing import Deque, Optional

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils.common import _default_filepath_fn, _default_cache_check_fn


@functional_datapipe("in_memory_cache")
class InMemoryCacheHolderIterDataPipe(IterDataPipe):
    r"""
    Iterable DataPipe that stores elements from the source DataPipe in memory, up to a size limit (if given).
    This cache is FIFO - once the cache is full, further elements will not be added to the cache
    until the previous ones are yielded and popped off the cache.

    Args:
        source_dp: source DataPipe from which elements are read and stored in memory
        size: The maximum size (in megabytes) that this DataPipe can hold in memory. This defaults to unlimited.
    """
    size: Optional[int] = None
    idx: int

    def __init__(self, source_dp, size=None):
        self.source_dp = source_dp
        # cache size in MB
        if size is not None:
            self.size = size * 1024 * 1024
        self.cache: Optional[Deque] = None
        self.idx = 0

    def __iter__(self):
        if self.cache:
            for idx, data in enumerate(self.source_dp):
                if idx < self.idx:
                    yield data
                else:
                    break
            yield from self.cache
        else:
            # Local cache
            cache: Deque = deque()
            idx = 0
            for data in self.source_dp:
                cache.append(data)
                # Cache reaches limit
                if self.size is not None and sys.getsizeof(cache) > self.size:
                    cache.popleft()
                    idx += 1
                yield data
            self.cache = cache
            self.idx = idx

    def __len__(self):
        try:
            return len(self.source_dp)
        except TypeError:
            if self.cache:
                return self.idx + len(self.cache)
            else:
                raise TypeError(f"{type(self).__name__} instance doesn't have valid length until the cache is loaded.")


class _CacheOp:
    def __init__(self, cache_holder, fn_name):
        self.cache_holder = cache_holder
        self.fn_name = fn_name

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self.cache_holder


@functional_datapipe("on_disk_cache")
class OnDiskCacheHolderIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe,
        filepath_fn=_default_filepath_fn,
        mode: str = "wb",
        cache_check_fn=_default_cache_check_fn,
    ):
        self.source_datapipe = source_datapipe
        self.filepath_fn = filepath_fn
        self.mode = mode
        self.cache_check_fn = cache_check_fn
        self.ops = []

    # TODO: Whenever `IterDataPipe` has a new magic function
    # implemented, it's needed accordingly
    def __iter__(self):
        raise RuntimeError("Please call `end_caching()` before iteration.")

    def __add__(self, other_datapipe):
        raise RuntimeError("`OnDiskCacheHolder` doesn't support add operation")

    def __reduce_ex__(self):
        raise RuntimeError("Please call `end_caching()` before calling graph or serialization.")

    def __getattr__(self, name):
        op = _CacheOp(self, name)
        self.ops.append(op)
        return op

    def end_caching(self):
        #  dp = self.source_datapipe.map(fn=lambda d: (self.filepath_fn(d), d))
        dp = self.source_datapipe
        todo_dp, cached_dp = dp.demux(2, self.cache_check_fn)
        # Cached: keeps filepath
        cached_dp = cached_dp.map(fn=self.filepath_fn)

        for op in self.ops:
            todo_dp = getattr(todo_dp, op.fn_name)(*op.args, **op.kwargs)

        todo_dp = todo_dp.save_to_disk(mode=self.mode, filepath_fn=self.filepath_fn)
        # Return filenames
        return cached_dp.concat(todo_dp)
