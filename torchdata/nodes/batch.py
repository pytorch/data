# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

from torchdata.nodes.base_node import BaseNode, T


class Batcher(BaseNode[List[T]]):
    """Batcher node batches the data from the source node into batches of size batch_size.
    If the source node is exhausted, it will return the batch or raise StopIteration.
    If drop_last is True, the last batch will be dropped if it is smaller than batch_size.
    If drop_last is False, the last batch will be returned even if it is smaller than batch_size.

    Args:
        source (BaseNode[T]): The source node to batch the data from.
        batch_size (int): The size of the batch.
        drop_last (bool): Whether to drop the last batch if it is smaller than batch_size. Default is True.
    """

    SOURCE_KEY = "source"

    def __init__(self, source: BaseNode[T], batch_size: int, drop_last: bool = True):
        super().__init__()
        self.source = source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self.source.reset(initial_state[self.SOURCE_KEY])
        else:
            self.source.reset()

    def next(self) -> List[T]:
        batch: List[T] = []
        while len(batch) < self.batch_size:
            try:
                item = next(self.source)
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
