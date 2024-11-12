# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, List, Optional

from torchdata.nodes.base_node import BaseNode, T


class Batcher(BaseNode[List[T]]):
    def __init__(self, source: BaseNode[T], batch_size: int, drop_last: bool = True):
        self.source = source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[List[T]]:
        self._it = self.Iter(self, initial_state)
        return self._it

    def get_state(self) -> Dict[str, Any]:
        if self._it is None:
            iter(self)
        return self._it.get_state()

    class Iter(Iterator[List[T]]):
        SOURCE_KEY = "source"

        def __init__(self, parent, initial_state: Optional[Dict[str, Any]]):
            self.source = parent.source
            self.batch_size = parent.batch_size
            self.drop_last = parent.drop_last

            if initial_state is not None:
                self.source.load_state_dict(initial_state[self.SOURCE_KEY])

            self._it = iter(self.source)

        def __iter__(self):
            return self

        def __next__(self) -> List[T]:
            batch = []
            while len(batch) < self.batch_size:
                try:
                    item = next(self._it)
                except StopIteration:
                    break
                batch.append(item)
                if len(batch) == self.batch_size:
                    return batch

            if len(batch) == self.batch_size:
                return batch
            elif len(batch) and not self.drop_last:
                return batch
            else:
                raise StopIteration()

        def get_state(self) -> Dict[str, Any]:
            return {self.SOURCE_KEY: self.source.state_dict()}
