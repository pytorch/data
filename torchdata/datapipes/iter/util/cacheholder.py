# Copyright (c) Facebook, Inc. and its affiliates.
import os.path
import sys

from collections import deque
from typing import Deque, List, Optional

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils.common import _default_filepath_fn


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

    def __init__(self, source_dp, size=None) -> None:
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
    """
    `OnDiskCacheHolder` is a factory IterDataPipe that creates cached local file for the
    output from a sequence of DataPipe operations, which are normally performance bottleneck like
    Download, Decompress. Default `filepath_fn` return a path in temporary directory based
    on the basename of data yielded from `source_datapipe`.

    Use `end_caching` method to stop tracing the sequence of Data operations and start saving
    result to local file system.

    Args:
        source_datapipe: DataPipe with URLs or file strings
        filepath_fn: Given URL or file path string, returns a file path to local file system
        extra_check_fn: Function to check if the traced operations need to be applied on
            the data from `source_datapipe`  with the file path returned by `filepath_fn`
        mode: mode in which the file will be opened for write the data

    Returns:
        It would returns a IterDataPipe that yields local file paths

    Example:
        url = IterableWrapper(["https://path/to/filename", ])
        cache_dp = file_dp.on_disk_cache(mode="wt").open_url().map(fn=lambda x: b''.join(x).decode(), input_col=1).end_caching()
        # Return a DataPipe yielding ["/tmp/xxx/filename", ]
    """
    def __init__(
        self,
        source_datapipe,
        filepath_fn=_default_filepath_fn,
        extra_check_fn=None,
        mode: str = "wb",
    ):
        self.source_datapipe = source_datapipe
        self.filepath_fn = filepath_fn
        self.extra_check_fn = extra_check_fn
        self.mode = mode
        self.ops: List[_CacheOp] = []

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

    def _cache_check_fn(self, data):
        filepath = self.filepath_fn(data)
        if self.extra_check_fn:
            return os.path.exists(filepath) and self.extra_check_fn(filepath)
        return os.path.exists(filepath)

    def end_caching(self):
        todo_dp, cached_dp = self.source_datapipe.demux(2, self._cache_check_fn)
        # Cached: keeps filepath
        cached_dp = cached_dp.map(fn=self.filepath_fn)

        for op in self.ops:
            todo_dp = getattr(todo_dp, op.fn_name)(*op.args, **op.kwargs)

        todo_dp = todo_dp.save_to_disk(mode=self.mode, filepath_fn=self.filepath_fn)
        # Return filenames
        return cached_dp.concat(todo_dp)
