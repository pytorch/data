# Copyright (c) Facebook, Inc. and its affiliates.
import sys

from collections import deque
from os import path
from typing import Optional

from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter import FileLoader
from torchdata.datapipes.utils.common import _default_filepath_fn


@functional_datapipe('in_memory_cache')
class InMemoryCacheHolderIterDataPipe(IterDataPipe):
    size: Optional[int] = None
    done: bool
    idx: int

    def __init__(self, source_dp, size=None):
        self.source_dp = source_dp
        # cache size in MB
        if size is not None:
            self.size = size * 1024 * 1024
        self.cache = None
        self.done = False
        self.idx = 0

    def __iter__(self):
        if self.done:
            for idx, data in enumerate(self.source_dp):
                if idx < self.idx:
                    yield data
                else:
                    break
            yield from self.cache
        else:
            # Local cache
            cache = deque()
            idx = 0
            for data in self.source_dp:
                cache.append(data)
                # Cache reaches limit
                if self.size is not None and \
                        sys.getsizeof(cache) > self.size:
                    cache.popleft()
                    idx += 1
                yield data
            self.done = True
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


@functional_datapipe('on_disk_cache')
class OnDiskCacheHolderIterDataPipe(IterDataPipe):
    """
        `OnDiskCacheHolder` is a factory DataPipe to create cached local file
        for the output of opDataPipe, which is normally performance
        bottleneck like Download, Decompress.
        Default `filepath_fn` return a path in temporary directory based
        on the basename of data yielded from `source_datapipe`.
        Example:
            from urllib.parse import urlparse

            def url_path_fn(url):
                return "~/.data/" + url.stripe().split('/')[-1]

            dp = ListofUrl(urls).on_disk_cache(HTTPReader, filepath_fn=url_path_fn)
            # Existing file will be skipped for downloading and directly loaded from disk
            # Non-existing file will be downloaded by HTTP request and saved to disk, then loaded from disk
    """
    def __new__(self,
                source_datapipe,
                opDataPipe,
                op_args=None,
                op_kwargs=None,
                op_map=None,
                mode='wb',
                filepath_fn=_default_filepath_fn):

        assert isinstance(source_datapipe, IterDataPipe), "'source_datapipe' needs to be an IterDataPipe"
        if op_args is None:
            op_args = ()
        if op_kwargs is None:
            op_kwargs = {}
        # source_datapipe should either yield url or file string
        dp = source_datapipe.map(fn=lambda data: (filepath_fn(data), data))
        # Out of order
        cached_dp, todo_dp = dp.demux(2, lambda d: 0 if path.exists(d[0]) else 1)

        # Cached: keeps filepath
        cached_dp = cached_dp.map(fn=lambda data: data[0])

        # Non-cached
        # opDataPipe yield bytes
        if not op_map:
            todo_dp = opDataPipe(todo_dp.map(fn=lambda data: data[1]), *op_args, **op_kwargs).save_to_disk(mode=mode, filepath_fn=filepath_fn)
        else:
            todo_dp = opDataPipe(todo_dp.map(fn=lambda data: data[1]), *op_args, **op_kwargs).map(op_map).save_to_disk(mode=mode, filepath_fn=filepath_fn)


        # Load files from local disk
        return FileLoader(cached_dp.concat(todo_dp))

