# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import hashlib
import inspect
import os.path
import sys
import time
import uuid
import warnings

from collections import deque
from functools import partial
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, TypeVar

try:
    import portalocker
except ImportError as e:
    if os.name == "nt" and str(e).startswith("DLL load failed while importing"):
        print(
            "Please take a look at FAQ in https://github.com/pytorch/data#frequently-asked-questions-faq"
            "for the solution of this Error."
        )
    raise


from torch.utils.data.datapipes.utils.common import _check_unpickable_fn, DILL_AVAILABLE

from torch.utils.data.graph import traverse
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileLister, IterDataPipe

if DILL_AVAILABLE:
    import dill

    dill.extend(use_dill=False)

T_co = TypeVar("T_co", covariant=True)

PROMISE_FILE_DELETE_TIMEOUT = 30
PROMISE_FILE_DELETE_RETRY_INTERVAL = 0.005


@functional_datapipe("in_memory_cache")
class InMemoryCacheHolderIterDataPipe(IterDataPipe[T_co]):
    r"""
    Stores elements from the source DataPipe in memory, up to a size limit
    if specified (functional name: ``in_memory_cache``). This cache is FIFO - once the cache is full,
    further elements will not be added to the cache until the previous ones are yielded and popped off from the cache.

    Args:
        source_dp: source DataPipe from which elements are read and stored in memory
        size: The maximum size (in megabytes) that this DataPipe can hold in memory. This defaults to unlimited.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> cache_dp = source_dp.in_memory_cache(size=5)
        >>> list(cache_dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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


def _hash_check(filepath, hash_dict, hash_type):

    if filepath not in hash_dict:
        return False

    if hash_type == "sha256":
        hash_func = hashlib.sha256()
    else:
        hash_func = hashlib.md5()

    # with portalocker.Lock(filepath, "rb", flags=portalocker.LockFlags.SHARED) as f:
    # TODO(VitalyFedyunin): Line above will require all readers (Win) to obtain proper locks,
    # I'm putting it on hold as we need to modify PyTorch core codebase heavily.
    with open(filepath, "rb") as f:
        chunk = f.read(1024 ** 2)
        while chunk:
            hash_func.update(chunk)
            chunk = f.read(1024 ** 2)

    return hash_func.hexdigest() == hash_dict[filepath]


def _promise_filename(filename):
    return filename + ".promise"


@functional_datapipe("on_disk_cache")
class OnDiskCacheHolderIterDataPipe(IterDataPipe):
    """
    Caches the outputs of multiple DataPipe operations to local files, which are
    typically performance bottleneck such download, decompress, and etc (functional name: ``on_disk_cache``).

    Must use ``.end_caching()`` to stop tracing the sequence of DataPipe operations and save the results to local files.

    Args:
        source_datapipe: IterDataPipe
        filepath_fn: Given data from ``source_datapipe``, returns file path(s) on local file system.
            Single file path is only allowed as output of the function.
            If resulted file name is different from the filename generated by the filename function of the end_cache
            original file name used to store list of yild files (and as cached items availability check)
        # TODO: Drop below as replacable by pipeline (?)
        hash_dict: A Dictionary mapping file names to their corresponding hashes. If ``hash_dict`` is specified,
            the extra hash check will be attached before saving data to local file system. If the data
            doesn't meet the hash, the pipeline will raise an Error.
        hash_type: The type of hash function to apply
        # TODO: Drop below as unused and replacable by pipeline
        extra_check_fn: Optional function to carry out extra validation on
            the given file path from ``filepath_fn``.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, HttpReader
        >>> url = IterableWrapper(["https://path/to/filename", ])
        >>> def _filepath_fn(url):
        >>>     temp_dir = tempfile.gettempdir()
        >>>     return os.path.join(temp_dir, os.path.basename(url))
        >>> hash_dict = {"expected_filepath": expected_MD5_hash}
        >>> cache_dp = url.on_disk_cache(filepath_fn=_filepath_fn, hash_dict=_hash_dict, hash_type="md5")
        >>> # You must call ``.end_caching`` at a later point to stop tracing and save the results to local files.
        >>> cache_dp = HttpReader(cache_dp).end_caching(mode="wb", filepath_fn=_filepath_fn)
    """

    _temp_dict: Dict = {}

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        filepath_fn: Optional[Callable] = None,
        hash_dict: Dict[str, str] = None,
        hash_type: str = "sha256",
        extra_check_fn: Optional[Callable[[str], bool]] = None,
    ):
        self.source_datapipe = source_datapipe

        if filepath_fn is not None:
            _check_unpickable_fn(filepath_fn)
        assert not inspect.isgeneratorfunction(filepath_fn)

        if hash_dict is not None and hash_type not in ("sha256", "md5"):
            raise ValueError("Invalid hash_type requested, should be one of {}".format(("sha256", "md5")))
        OnDiskCacheHolderIterDataPipe._temp_dict[self] = (filepath_fn, hash_dict, hash_type, extra_check_fn)

        self._end_caching_flag: bool = False

    def __iter__(self):
        if self._end_caching_flag:
            yield from self.source_datapipe
        else:
            # In case of BC breaking, use RuntimeError for now. Warning is another option
            raise RuntimeError("Please call `end_caching()` before iteration.")

    def __add__(self, other_datapipe):
        raise RuntimeError("`OnDiskCacheHolder` doesn't support add operation")

    # Since Demux is using this function, we should not attach it to OnDiskCacheHolder instance.
    # Otherwise, it would cause infinite recursion in graph traversal
    @staticmethod
    def _cache_check_fn(data, filepath_fn, hash_dict, hash_type, extra_check_fn):
        filepath = data if filepath_fn is None else filepath_fn(data)
        result = 1
        assert not isinstance(filepath, (list, tuple))  # BC breaking, now only str is accepted as return

        cached_file_exists = True
        if os.path.exists(_get_list_filename(filepath)):
            return 2
        if not os.path.exists(filepath):
            cached_file_exists = False
        elif hash_dict is not None and not _hash_check(filepath, hash_dict, hash_type):
            # TODO: It is safer to assume that entire cache is compromised and require user to wipe it
            cached_file_exists = False
        elif extra_check_fn is not None and not extra_check_fn(filepath):
            # TODO: It is safer to assume that entire cache is compromised and require user to wipe it
            cached_file_exists = False
        print(f"check result for {filepath} is {cached_file_exists}")
        if not cached_file_exists:
            promise_filepath = _promise_filename(filepath)
            dirname = os.path.dirname(promise_filepath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            with portalocker.Lock(promise_filepath, "a+", flags=portalocker.LockFlags.EXCLUSIVE) as promise_fh:
                promise_fh.seek(0)
                data = promise_fh.read()
                # TODO(VitalyFedyunin): Potentially there is old .promise file from previous failed run, we
                # need to somehow propagate uniq session id for dataloader, save and compare it here,
                # raising error
                file_exists = len(data) > 0
                if not file_exists:
                    result = 0
                    promise_fh.seek(0)
                    promise_fh.write("[dataloader session uid]")
                    promise_fh.truncate()
                    promise_fh.flush()

        return result

    def _end_caching(self):
        filepath_fn, hash_dict, hash_type, extra_check_fn = OnDiskCacheHolderIterDataPipe._temp_dict.pop(self)

        todo_dp, cached_dp, one_many_cached_dp = self.source_datapipe.demux(
            3,
            partial(
                OnDiskCacheHolderIterDataPipe._cache_check_fn,
                filepath_fn=filepath_fn,
                hash_dict=hash_dict,
                hash_type=hash_type,
                extra_check_fn=extra_check_fn,
            ),
        )
        # Cached: keep filepath(s)
        cached_dp = cached_dp.map(fn=filepath_fn)

        one_many_cached_dp = one_many_cached_dp.map(fn=filepath_fn)
        one_many_cached_dp = _ExtractFilesFromList(one_many_cached_dp)

        # Convert list back to single elements
        # cached_dp = cached_dp.unbatch(-1)

        self.source_datapipe = todo_dp.memory_cell()
        self._end_caching_flag = True
        return cached_dp, one_many_cached_dp


def _read_bytes(fd):
    return b"".join(fd)


def _read_str(fd):
    return "".join(fd)


def _find_promise_file(filename):
    promise_filename = _promise_filename(filename)
    # while not os.path.exists(promise_filename):
    #     dirname = os.path.dirname(promise_filename)
    #     if dirname == os.path.dirname(dirname):
    #         promise_filename = _promise_filename(filename)
    #         break
    #     promise_filename = _promise_filename(dirname)
    return promise_filename


def _is_promise_pending(promise_filename):
    return os.path.exists(promise_filename)


def _wait_promise_fn(timeout, filename):
    promise_filename = _find_promise_file(filename)
    start = time.time()
    while _is_promise_pending(promise_filename):
        time.sleep(0.01)
        if time.time() - start > timeout:
            raise Exception(
                f"OnDiskCache Exception: {filename} expected to be written by different process, "
                + f"but file is not ready in {timeout} seconds."
            )
    return filename


@functional_datapipe("memory_cell")
class _MemoryCellIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, remember_elements=10):
        self.source_datapipe = source_datapipe
        self.buffer = [None for i in range(remember_elements)]
        self.remember_elements = remember_elements
        self.buffer_pos = -1
        # TODO(VitalyFedyunin): Make it friendly to save/restore state

    def __iter__(self):
        for item in self.source_datapipe:
            item_id = uuid.uuid4()
            self.buffer_pos = (self.buffer_pos + 1) % self.remember_elements
            self.buffer[self.buffer_pos] = (item_id, item)
            yield item

    def get_last(self):
        # Returns tuple of elements, autogenerated id of the last returned row and its value
        return self.buffer[self.buffer_pos]

    def get_buffer(self):
        # Returns last returned id+element and others in the order from latest to oldest.
        result = []
        for i in range(self.remember_elements):
            idx = (self.buffer_pos - i) % self.remember_elements
            if self.buffer[idx] is not None:
                result.append(self.buffer[idx])
        return result


def _get_list_filename(file_name):
    return file_name + ".torchdata_list"


class _ExtractFilesFromList(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for filename in self.source_datapipe:
            print("Extracting files from list ", _get_list_filename(filename))
            with open(_get_list_filename(filename)) as fh:
                for line in fh:
                    inner_file_name = line.rstrip()
                    print("extracted file ", inner_file_name)
                    yield filename, inner_file_name


class _FulfilledPromisesIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, memory_cell_dp, first_filepath_fn):
        self.source_datapipe = source_datapipe
        self.memory_cell_dp = memory_cell_dp
        self.first_filepath_fn = first_filepath_fn

    @staticmethod
    def _del_promise_file(promise_filename, filename):
        print(f"Removing {promise_filename} for {filename}")
        if os.path.exists(promise_filename):
            retry = True
            start = time.time()
            while retry:
                retry = False
                try:
                    os.unlink(promise_filename)
                except Exception as e:
                    # Workaround about Windows not letting to delete file, while it is open by another process
                    retry = True
                    if time.time() - start > PROMISE_FILE_DELETE_TIMEOUT:
                        raise Exception("Timeout while trying to recover from the ", type(e), e)
                    time.sleep(PROMISE_FILE_DELETE_RETRY_INTERVAL)
        else:
            warnings.warn(
                f"Attempt to mark {promise_filename} promise (base of file {filename}) as fulfilled failed. Potentially missmatching filename functions of on_disk_cache and end_cache."
            )

    def __iter__(self):
        # old_promise_filename = None
        # old_filename = None
        # first_entry = True
        last_record_uuid = None
        # TODO(VitalyFedyunin): Limit buffer size here. It is only contains file names from archive,
        # but better be safe than sorry.
        # buffer: List[Any] = []
        one_to_many_detected = False
        one_to_one_detected = False

        def fulfill_old_promises(buffer, last_record_uuid, first_filepath_fn):
            for old_rec_uuid, old_rec in buffer:
                original_file_name = first_filepath_fn(old_rec)
                old_promise_filename = _find_promise_file(original_file_name)
                self._del_promise_file(old_promise_filename, original_file_name)
                if old_rec_uuid == last_record_uuid:
                    break

        for filename in self.source_datapipe:
            rec_uuid, record = self.memory_cell_dp.get_last()
            original_file_name = self.first_filepath_fn(record)
            if original_file_name != filename:
                one_to_many_detected = True
                if one_to_one_detected:
                    raise Exception("DIsovered different keys when one-to-one mode previously assumed")
                # We are dealing with one-to-many situation now
                with open(_get_list_filename(original_file_name), "a") as fh:
                    fh.write(f"{filename}\n")
            else:
                one_to_one_detected = True
                if one_to_many_detected:
                    # Keys should be always the same (1-1 situation) or always different (1-many) sutuation
                    raise Exception("first key somehow equal to secondary key")
            print(f"{filename} generateed from rec {rec_uuid} {record}")
            if rec_uuid != last_record_uuid:
                fulfill_old_promises(self.memory_cell_dp.get_buffer()[1:], last_record_uuid, self.first_filepath_fn)
                last_record_uuid = rec_uuid
            yield filename

        if last_record_uuid is not None:
            fulfill_old_promises(self.memory_cell_dp.get_buffer(), last_record_uuid, self.first_filepath_fn)


@functional_datapipe("end_caching")
class EndOnDiskCacheHolderIterDataPipe(IterDataPipe):
    """
    Indicates when the result of prior DataPipe will be saved local files specified
    by ``filepath_fn`` (functional name: ``end_caching``). Moreover, the result of source DataPipe
    is required to be a tuple of metadata and data, or a tuple of metadata and file handle.

    Args:
        datapipe: IterDataPipe with at least one ``OnDiskCacheHolder`` in the graph.
        mode: Mode in which the cached files are opened to write the data on disk. This is needed
            to be aligned with the type of data or file handle from ``datapipe``. ``"wb"`` is used by default.
        filepath_fn: Optional function to extract filepath from the metadata from ``datapipe``.
            By default, it would directly use the ?metadata? as file path.
        same_filepath_fn: Set to ``True`` to use same ``filepath_fn`` from the ``OnDiskCacheHolder``.
        skip_read: Boolean value to skip reading the file handle from ``datapipe``.
            By default, reading is enabled and reading function is created based on the ``mode``.
        timeout: Integer value of seconds to wait for uncached item to be written to disk

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, HttpReader
        >>> url = IterableWrapper(["https://path/to/filename", ])
        >>> def _filepath_fn(url):
        >>>     temp_dir = tempfile.gettempdir()
        >>>     return os.path.join(temp_dir, os.path.basename(url))
        >>> hash_dict = {"expected_filepath": expected_MD5_hash}
        >>> # You must call ``.on_disk_cache`` at some point before ``.end_caching``
        >>> cache_dp = url.on_disk_cache(filepath_fn=_filepath_fn, hash_dict=_hash_dict, hash_type="md5")
        >>> # You must call ``.end_caching`` at a later point to stop tracing and save the results to local files.
        >>> cache_dp = HttpReader(cache_dp).end_caching(mode="wb", filepath_fn=_filepath_fn)
    """

    def __new__(cls, datapipe, mode="wb", filepath_fn=None, *, same_filepath_fn=False, skip_read=False, timeout=300):
        if filepath_fn is not None and same_filepath_fn:
            raise ValueError("`filepath_fn` is mutually exclusive with `same_filepath_fn`")

        graph = traverse(datapipe, only_datapipe=True)
        # Get the last CacheHolder
        cache_holder = EndOnDiskCacheHolderIterDataPipe._recursive_search(graph)
        if cache_holder is None:
            raise RuntimeError("Expected `OnDiskCacheHolder` existing in pipeline when `end_caching` is invoked")
        if cache_holder._end_caching_flag:
            raise RuntimeError("`end_caching` can only be invoked once per `OnDiskCacheHolder`")

        first_filepath_fn, _hash_dict, _hash_type, _ = OnDiskCacheHolderIterDataPipe._temp_dict[cache_holder]
        cached_dp, one_many_cached_dp = cache_holder._end_caching()
        cached_dp = cached_dp.map(functools.partial(_wait_promise_fn, timeout))
        one_many_cached_dp = one_many_cached_dp.map(functools.partial(_wait_promise_fn, timeout), input_col=0)
        one_many_cached_dp = one_many_cached_dp.map(lambda x: x[1])
        # cached_dp = FileLister(cached_dp, recursive=True)

        memory_cell_dp = cache_holder.source_datapipe

        if same_filepath_fn:
            filepath_fn = first_filepath_fn

        todo_dp = datapipe
        if not skip_read:
            if "t" in mode:
                todo_dp = todo_dp.map(fn=_read_str, input_col=1)
            else:
                todo_dp = todo_dp.map(fn=_read_bytes, input_col=1)

        def dbg(prefix, data):
            print(prefix, data)
            return data

        if filepath_fn is not None:
            todo_dp = (
                todo_dp.map(partial(dbg, "original name"), input_col=0)
                .map(fn=filepath_fn, input_col=0)
                .map(partial(dbg, "generated name"), input_col=0)
            )

        # Extra hash check here when hash is provided.
        # And, raise Error if data returned from prior operations doesn't meet hash
        if _hash_dict is not None:
            todo_dp = todo_dp.check_hash(_hash_dict, _hash_type)

        todo_dp = todo_dp.save_to_disk(mode=mode)
        todo_dp = _FulfilledPromisesIterDataPipe(todo_dp, memory_cell_dp, first_filepath_fn)

        # TODO(VitalyFedyunin): This impacts determinism for partial cache situations
        return todo_dp.concat(cached_dp).concat(one_many_cached_dp)

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
