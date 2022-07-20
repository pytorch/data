# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import shutil
import sys
import urllib.parse
from typing import Any, Dict, Iterator, Tuple

from torch.utils.data.datapipes.utils.common import StreamWrapper

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


def cache_by_fname(s):
    result = re.sub("^.*/", "", s)
    return urllib.parse.quote(result)


if os.name == "nt":
    default_cachedir = "datacache"
else:
    default_cachedir = "_datacache"


@functional_datapipe("filecache")
class FileCacheIterDataPipe(IterDataPipe[Dict]):
    r""" """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[str, Any]],
        cachedir=default_cachedir,
        cachename=cache_by_fname,
        chunksize=1024**2,
        verbose=False,
        makedir=True,
    ) -> None:
        super().__init__()
        if not os.path.exists(cachedir):
            if makedir:
                os.makedirs(cachedir)
            else:
                raise ValueError(f"Cache directory {cachedir} does not exist.")
        self.source_datapipe: IterDataPipe[Tuple[str, Any]] = source_datapipe
        self.cachedir = cachedir
        self.cachename = cachename
        self.verbose = verbose
        self.chunksize = chunksize

    def __iter__(self) -> Iterator[Dict]:
        for url, stream in self.source_datapipe:
            cached = os.path.join(self.cachedir, self.cachename(url))
            if not os.path.exists(cached):
                if self.verbose:
                    print(f"# downloading {url} to {cached}", file=sys.stderr)
                with open(cached + ".temp", "wb") as dest:
                    shutil.copyfileobj(stream, dest, self.chunksize)
                os.rename(cached + ".temp", cached)
            if self.verbose:
                print(f"# returning {cached}", file=sys.stderr)
            cached_stream = open(cached, "rb")
            yield url, StreamWrapper(cached_stream)

    def __len__(self) -> int:
        return len(self.source_datapipe)
