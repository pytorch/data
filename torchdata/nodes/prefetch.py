# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator

from torchdata.nodes import BaseNode, T

from torchdata.nodes.map import _SingleThreadedMapper

from ._populate_queue import _populate_queue


class Prefetcher(BaseNode[T]):
    def __init__(self, source: BaseNode[T], prefetch_factor: int):
        self.source = source
        self.prefetch_factor = prefetch_factor

    def iterator(self) -> Iterator[T]:
        return _SingleThreadedMapper(
            source=self.source,
            prefetch_factor=self.prefetch_factor,
            worker=_populate_queue,
        )
