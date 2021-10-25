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
    `OnDiskCacheHolder` is a IterDataPipe that caches output of multiple DataPipe operations
    to local files, which are normally performance bottleneck like download, decompress,
    and etc.

    Use `end_caching()` to stop tracing the sequence of DataPipe operations and start saving
    result to a local file specified by `filepath_fn` per iteration. And, the result of these
    operations is required to be a tuple of file path and data.

    Args:
        source_datapipe: DataPipe with URLs or file strings
        filepath_fn: Given URL or file path string, returns a file path to local file system. As default,
            a file path in a temporary directory with basename of the given URL or file path is returned
        extra_check_fn: Given the file path returned by `filepath_fn`, returns if the traced DataPipe
            operation can be skipped as the result has been correctly cached. By default, it will check if
            the cached file exists in local file system. Hash check can be specified by this function.
        mode: Mode in which cached files are opened for write the data. Binary mode by default.

    Returns:
        An IterDataPipe that yields local file paths

    Example:
        url = IterableWrapper(["https://path/to/filename", ])
        cache_dp = url.on_disk_cache().open_url().map(fn=lambda x: b''.join(x), input_col=1).end_caching()

        # Hash check
        def hash_fn(filepath):
            hash_fn = hashlib.md5()
            with open(filepath, "rb") as f:
                chunk = f.read(1024 ** 2)
                while chunk:
                    hash_fn.update(chunk)
                    chunk = f.read(1024 ** 2)
            return hash_fn.hexdigest() == expected_MD5_hash
        cache_dp = url.on_disk_cache(extra_check_fn=hash_fn).open_url().map(fn=lambda x: b''.join(x), input_col=1).end_caching()
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
        # TODO: Figure out how many operations can not be traced
        if name == "on_disk_cache":
            raise RuntimeError("`OnDiskCacheHolder` doesn't support", name, "operation during tracing")
        op = _CacheOp(self, name)
        self.ops.append(op)
        return op

    def _cache_check_fn(self, data):
        filepath = self.filepath_fn(data)
        if not os.path.exists(filepath):
            return False
        if self.extra_check_fn:
            return self.extra_check_fn(filepath)
        return True

    def end_caching(self):
        todo_dp, cached_dp = self.source_datapipe.demux(2, self._cache_check_fn)
        # Cached: keeps filepath
        cached_dp = cached_dp.map(fn=self.filepath_fn)

        for op in self.ops:
            todo_dp = getattr(todo_dp, op.fn_name)(*op.args, **op.kwargs)

        todo_dp = todo_dp.save_to_disk(mode=self.mode, filepath_fn=self.filepath_fn)
        # Return filenames
        return cached_dp.concat(todo_dp)
