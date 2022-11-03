# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


def _return_zero(_):
    return 0


@functional_datapipe("group_batch")
class GroupBatcherIterDataPipe(IterDataPipe):
    """
    dp = IterableWrapper(range(20))
    dp = dp.bucket_batch(batch_size = 2, number_of_groups = 5, group_key_fn = lambda x: x % 5)
    list(dp)
    [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [10, 15], [11, 16], [12, 17], [13, 18], [14, 19]]
    """

    def __init__(self, source_datapipe, batch_size: int, number_of_groups: int = 1, group_key_fn=_return_zero):
        self.batch_size = batch_size
        self.source_datapipe = source_datapipe
        self._number_of_groups = number_of_groups
        self._group_key_fn = group_key_fn
        self._groups_buffer = [[] for i in range(self._number_of_groups)]

    def __iter__(self):
        self._groups_buffer = [[] for i in range(self._number_of_groups)]
        for element in self.source_datapipe:
            group_idx = self._group_key_fn(element)
            # TODO: This operation can be optimized by preallocating buffers of batch_size and tracking number of elements
            self._groups_buffer[group_idx].append(element)
            if len(self._groups_buffer[group_idx]) == self.batch_size:
                result = self._groups_buffer[group_idx]
                self._groups_buffer[group_idx] = []
                yield result

    def reset(self) -> None:
        self._groups_buffer = []
