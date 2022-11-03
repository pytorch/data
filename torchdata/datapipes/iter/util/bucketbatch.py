# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


def _return_zero(_):
    return 0


@functional_datapipe("bucket_batch")
class BucketBatchIterDataPipe(IterDataPipe):
    """
    dp = IterableWrapper(range(20))
    dp = dp.bucket_batch(batch_size = 2, number_of_buckets = 5, bucket_function = lambda x: x % 5)
    list(dp)
    [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [10, 15], [11, 16], [12, 17], [13, 18], [14, 19]]
    """

    def __init__(self, source_datapipe, batch_size: int, number_of_buckets: int = 1, bucket_function=_return_zero):
        self.batch_size = batch_size
        self.source_datapipe = source_datapipe
        self._number_of_buckets = number_of_buckets
        self._bucket_function = bucket_function
        self._buckets_buffer = [[] for i in range(self._number_of_buckets)]

    def __iter__(self):
        self._buckets_buffer = [[] for i in range(self._number_of_buckets)]
        for element in self.source_datapipe:
            bucket_idx = self._bucket_function(element)
            # TODO: This operation can be optimized by preallocating buffers of batch_size and tracking number of elements
            self._buckets_buffer[bucket_idx].append(element)
            if len(self._buckets_buffer[bucket_idx]) == self.batch_size:
                result = self._buckets_buffer[bucket_idx]
                self._buckets_buffer[bucket_idx] = []
                yield result

    def reset(self) -> None:
        self._buckets_buffer = []
