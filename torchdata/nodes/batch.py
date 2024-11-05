# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, List, Optional

from torchdata.nodes.base_node import BaseNode, T


class Batcher(BaseNode[List[T]]):
    SOURCE_KEY = "source"

    def __init__(self, source: BaseNode[T], batch_size: int, drop_last: bool = True):
        self.source = source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[List[T]]:
        if initial_state is not None:
            self.source.load_state_dict(initial_state[self.SOURCE_KEY])

        batch = []
        for item in self.source:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) and not self.drop_last:
            yield batch

    def get_state(self) -> Dict[str, Any]:
        return {self.SOURCE_KEY: self.source.state_dict()}
