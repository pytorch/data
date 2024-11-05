# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, Optional

from torchdata.nodes import BaseNode, T

from torchdata.nodes.map import _SingleThreadedMapper

from ._populate_queue import _populate_queue


class Prefetcher(BaseNode[T]):
    def __init__(self, source: BaseNode[T], prefetch_factor: int, snapshot_frequency: int = 0):
        self.source = source
        self.prefetch_factor = prefetch_factor
        self.snapshot_frequency = snapshot_frequency
        self._it: Optional[_SingleThreadedMapper] = None

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        self._it = _SingleThreadedMapper(
            source=self.source,
            prefetch_factor=self.prefetch_factor,
            worker=_populate_queue,
            snapshot_frequency=self.snapshot_frequency,
            initial_state=initial_state,
        )
        return self._it

    def get_state(self) -> Dict[str, Any]:
        assert self._it is not None
        return self._it.get_state()
