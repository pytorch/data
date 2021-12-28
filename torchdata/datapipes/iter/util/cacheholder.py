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
        source_datapipe: IterDataPipe
        filepath_fn: Given data from `source_datapipe`, returns file path(s) on local file system.
            Single file path, tuple or list of file paths is accepted as return type.
            And, generator function that yields file paths is also allowed.
            As default, data from `source_datapipe` is directly used to determine the existency of cache.
        hash_dict: Dict mapping file names to their corresponding hashes. If hash_dict is specified,
            the extra hash check will be attached before saving data to local file system. If the data
            doesn't meet the hash, the pipeline will raise Error.
        hash_type: The type of hash function to apply
        extra_check_fn: Optional function to carry out extra validation on
            the given file path from `filepath_fn`.

    Example:
        url = IterableWrapper(["https://path/to/filename", ])

        def _filepath_fn(url):
            temp_dir = tempfile.gettempdir()
            return os.path.join(temp_dir, os.path.basename(url))

        hash_dict = {"expected_filepaht": expected_MD5_hash}

        cache_dp = url.on_disk_cache(filepath_fn=_filepath_fn, hash_dict=_hash_dict, hash_type="md5")
        cache_dp = HttpReader(cache_dp).end_caching(mode="wb". filepath_fn=_filepath_fn)
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        filepath_fn: Optional[Callable] = None,
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

    def _cache_check_fn(self, data) -> bool:
        filepaths = data if self.filepath_fn is None else self.filepath_fn(data)
        if not isinstance(filepaths, (list, tuple)):
            filepaths = [
                filepaths,
            ]

        for filepath in filepaths:
            if not os.path.exists(filepath):
                return False

            if self.hash_dict is not None and not self._hash_check(filepath):
                return False

            if self.extra_check_fn is not None and not self.extra_check_fn(filepath):
                return False

        return True

    def _hash_check(self, filepath) -> bool:

        if filepath not in self.hash_dict:  # type: ignore[operator]
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

        return hash_func.hexdigest() == self.hash_dict[filepath]  # type: ignore[index]

    def _end_caching(self) -> IterDataPipe:
        todo_dp, cached_dp = self.source_datapipe.demux(2, self._cache_check_fn)
        self._end_caching_flag = True
        # Re-assign source_datapipe
        # self.source_datapipe = todo_dp  # TODO: This causes infinite recursion in graph traversal

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
    `EndOnDiskCacheHolder` is a IterDataPipe that indicates when the result of
    prior DataPipe will be saved local files specified by `filepath_fn`
    And, the result of source DataPipe is required to be a tuple of metadata and data,
    or a tuple of metadata and file handle.

    Args:
        datapipe: IterDataPipe with at least one `OnDiskCacheHolder` in the graph.
        mode: Mode in which cached files are opened for write the data. This is needed
            to be aligned with the type of data or file handle from `datapipe`.
        filepath_fn: Optional function to extract filepath from the metadata from `datapipe`.
            As default, it would directly use the metadata as file path.
        same_filepath_fn: Set to `True` to use same `filepath_fn` from the `OnDiskCacheHolder`.
        skip_read: Boolean value to skip reading the file handle from `datapipe`.
            As default, reading is enabled and reading function is created based on the `mode`.
    """

    def __new__(cls, datapipe, mode="w", filepath_fn=None, *, same_filepath_fn=False, skip_read=False):
        if filepath_fn is not None and same_filepath_fn:
            raise ValueError("`filepath_fn` is mutually exclusive with `same_filepath_fn`")

        print("-------About to traverse graph in end_caching.--------")
        graph = traverse(datapipe, exclude_primitive=True)
        print("*******Graph is succesfully traversed in end_caching.******")
        # Get the last CacheHolder
        cache_holder = EndOnDiskCacheHolderIterDataPipe._recursive_search(graph)
        print("*******Recursive search is succesful in end_caching.******")
        if cache_holder is None:
            raise RuntimeError("Expected `OnDiskCacheHolder` existing in pipeline when `end_caching` is invoked")
        if cache_holder._end_caching_flag:
            raise RuntimeError("`end_caching` can only be invoked once per `OnDiskCacheHolder`")
        print(f"cache_holder: {cache_holder}")

        cached_dp = cache_holder._end_caching()

        if same_filepath_fn:
            filepath_fn = cache_holder.filepath_fn

        todo_dp = datapipe
        if not skip_read:
            if "b" in mode:
                todo_dp = todo_dp.map(fn=_read_bytes, input_col=1)
            else:
                todo_dp = todo_dp.map(fn=_read_str, input_col=1)

        todo_dp = todo_dp.map(fn=filepath_fn, input_col=0)

        # Extra hash check here when hash is provided.
        # And, raise Error if data returned from prior operations doesn't meet hash
        if cache_holder.hash_dict is not None:
            todo_dp = todo_dp.check_hash(cache_holder.hash_dict, cache_holder.hash_type)

        todo_dp = todo_dp.save_to_disk(mode=mode)

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
