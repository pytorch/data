# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Dict, Iterator

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


def shardexpand(s):
    expansion = r"[{][0-9]+[.][.][0-9]+[}]"
    m = re.search(expansion, s)
    if not m:
        return [s]
    prefix = s[: m.start()]
    rest = shardexpand(s[m.end() :])
    rng = s[m.start() + 1 : m.end() - 1]
    lohi = rng.split("..")
    if len(lohi[0]) == len(lohi[1]) and lohi[0].startswith("0"):
        fmt = "{prefix}{i:0>{l}d}{r}"
    elif len(lohi[0]) <= len(lohi[1]):
        if lohi[0].startswith("0") and lohi[0] != "0":
            raise ValueError("shardexpand: low bound must not start with 0 if low bound is shorter")
        fmt = "{prefix}{i}{r}"
    else:
        raise ValueError("shardexpand: low bound must be shorter than high bound")
    lo, hi = [int(x) for x in lohi]
    if lo >= hi:
        raise ValueError(f"shardexpand: bad range in in shard spec {s}.")
    result = []
    for i in range(lo, hi + 1):
        for r in rest:
            expanded = fmt.format(prefix=prefix, i=i, r=r, l=len(lohi[1]))
            result.append(expanded)
    return result


@functional_datapipe("shardexpand")
class ShardExpanderIterDataPipe(IterDataPipe[Dict]):
    r"""
    Expands incoming shard strings into shards.

    Sharded data files are named using shell-like brace notation. For example,
    an ImageNet dataset sharded into 1200 shards and stored on a web server
    might be named `imagenet-{000000..001199}.tar`.

    Note that shard names can be expanded without any server transactions;
    this makes `shardexpand` reproducible and storage system independent
    (unlike `ListFiles` etc.).

    Args:
        source_datapipe: a DataPipe yielding a stream of  pairs

    Returns:
        a DataPipe yielding a stream of expanded pathnames.
    """

    def __init__(self, source_datapipe: IterDataPipe[str]) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[str] = source_datapipe

    def __iter__(self) -> Iterator[str]:
        for path in self.source_datapipe:
            for expanded in shardexpand(path):
                yield expanded

    def __len__(self) -> int:
        return len(self.source_datapipe)
