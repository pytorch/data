# Copyright (c) Facebook, Inc. and its affiliates.
import hashlib
import inspect
import os.path
import sys

from collections import deque
from typing import Callable, Deque, Dict, Iterator, Optional, TypeVar

from torch.utils.data.graph import traverse
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("in_memory_cache")
class InMemoryCacheHolderIterDataPipe(IterDataPipe[T_co]):
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

    def __init__(self, source_dp: IterDataPipe[T_co], size: Optional[int] = None) -> None:
        self.source_dp: IterDataPipe[T_co] = source_dp
        # cache size in MB
        if size is not None:
            self.size = size * 1024 * 1024
        self.cache: Optional[Deque] = None
        self.idx: int = 0

    def __iter__(self) -> Iterator[T_co]:
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

    def __len__(self) -> int:
        try:
            return len(self.source_dp)
        except TypeError:
            if self.cache:
                return self.idx + len(self.cache)
            else:
                raise TypeError(f"{type(self).__name__} instance doesn't have valid length until the cache is loaded.")


def _generator_to_list(gen_fn):
    def list_fn(*args, **kwargs):
        gen = gen_fn(*args, **kwargs)
        return list(gen)
    return list_fn


@functional_datapipe("on_disk_cache")
class OnDiskCacheHolderIterDataPipe(IterDataPipe):
    """
    `OnDiskCacheHolder` is a IterDataPipe that caches output of multiple DataPipe operations
    to local files, which are normally performance bottleneck like download, decompress,
    and etc.

    Use `.end_caching()` to stop tracing the sequence of DataPipe operations and save result to local files.

    Args:
        source_datapipe: DataPipe with URLs or file strings
        filepath_fn: Given URL or file path string, returns a file path to local file system. As default,
            a file path in a temporary directory with basename of the given URL or file path is returned
        extra_check_fn: Given the file path returned by `filepath_fn`, returns if the traced DataPipe
            operation can be skipped as the result has been correctly cached. By default, it will check if
            the cached file exists in local file system. Hash check can be specified by this function.

    Example:
        url = IterableWrapper(["https://path/to/filename", ])
        cache_dp = url.on_disk_cache().open_url().map(fn=lambda x: b''.join(x), input_col=1).end_caching()

        # Cache with Hash check
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
        source_datapipe: IterDataPipe,
        filepath_fn,
        hash_dict: Dict[str, str] = None,
        hash_type: str = "sha256",
        extra_check_fn: Optional[Callable[[str], bool]] = None,
    ):
        self.source_datapipe = source_datapipe
        self.filepath_fn = _generator_to_list(filepath_fn) if inspect.isgeneratorfunction(filepath_fn) else filepath_fn
        self.hash_dict = hash_dict
        if hash_dict is not None and hash_type not in ("sha256", "md5"):
            raise ValueError("Invalid hash_type requested, should be one of {}".format(("sha256", "md5")))

        self.hash_type = hash_type
        self.extra_check_fn = extra_check_fn
        self._end_caching_flag: bool = False

    def __iter__(self):
        if self._end_caching_flag:
            for d in self.source_datapipe:
                yield d
        else:
            # In case of BC breaking, use RuntimeError for now. Warning is another option
            raise RuntimeError("Please call `end_caching()` before iteration.")

    def __add__(self, other_datapipe):
        raise RuntimeError("`OnDiskCacheHolder` doesn't support add operation")

    def _cache_check_fn(self, data):
        filepaths = self.filepath_fn(data)
        if isinstance(filepaths, str):
            filepaths = [filepaths, ]

        for filepath in filepaths:
            if not os.path.exists(filepath):
                return False

            if self.hash_dict is not None and not self._hash_check(filepath):
                return False

            if self.extra_check_fn is not None and not self.extra_check_fn(filepath):
                return False

        return True

    def _hash_check(self, filepath):

        if filepath not in self.hash_dict:
            return False

        if self.hash_type == "sha256":
            hash_func = hashlib.sha256()
        else:
            hash_func = hashlib.md5()

        with open(filepath, "rb") as f:
            chunk = f.read(1024 ** 2)
            while chunk:
                hash_func.update(chunk)
                chunk = f.read(1024 ** 2)

        return hash_func.hexdigest() == self.hash_dict[filepath]

    def _end_caching(self):
        todo_dp, cached_dp = self.source_datapipe.demux(2, self._cache_check_fn)
        self._end_caching_flag = True
        # Re-assign source_datapipe
        self.source_datapipe = todo_dp

        # Cached: keep filepath(s)
        cached_dp = cached_dp.map(fn=self.filepath_fn)
        # Convert list back to single elements
        return cached_dp.unbatch(-1)


def _read_bytes(fd):
    return b"".join(fd)


def _read_str(fd):
    return "".join(fd)


@functional_datapipe("end_caching")
class EndOnDiskCacheHolderIterDataPipe(IterDataPipe):
    """
    `EndOnDiskCacheHolder` is a IterDataPipe that indicates when theresult of
    prior DataPipe will be saved local files specified by `filepath_fn` of
    the corresponding `OnDiskCacheHolder`. And, the result is required to be
    a tuple of file path and data.

    Args:
        mode: Mode in which cached files are opened for write the data. Binary mode by default.
    """

    def __new__(cls, datapipe, mode="w", filepath_fn=None):
        graph = traverse(datapipe, exclude_primitive=True)
        cache_holder = EndOnDiskCacheHolderIterDataPipe._recursive_search(graph)
        if cache_holder is None:
            raise RuntimeError(
                "Expected `OnDiskCacheHolder` existing in pipeline when `end_caching` is invoked"
            )
        if cache_holder._end_caching_flag:
            raise RuntimeError("`end_caching` can only be invoked once per `OnDiskCacheHolder`")

        cached_dp = cache_holder._end_caching()
        if "b" in mode:
            todo_dp = datapipe.map(fn=_read_bytes, input_col=1)
        else:
            todo_dp = datapipe.map(fn=_read_str, input_col=1)
        todo_dp = todo_dp.save_to_disk(mode=mode, filepath_fn=filepath_fn)
        return cached_dp.concat(todo_dp)

    @staticmethod
    def _recursive_search(graph):
        for dp in graph.keys():
            # Find the closest CacheHolder
            if isinstance(dp, OnDiskCacheHolderIterDataPipe):
                return dp
        for dp in graph.values():
            res = EndOnDiskCacheHolderIterDataPipe._recursive_search(dp)
            if res is not None:
                return res
        return None
