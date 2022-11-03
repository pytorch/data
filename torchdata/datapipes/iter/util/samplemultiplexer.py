# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, Iterator, Optional, Sized, TypeVar

from torchdata.datapipes.iter import IterDataPipe


T_co = TypeVar("T_co", covariant=True)


class SampleMultiplexerDataPipe(IterDataPipe[T_co]):
    """
    Takes a `Dict` of (IterDataPipe, Weight), and yields items by sampling from these
    DataPipes with respect to their weights. When individual DataPipes are exhausted, continues to sample from
    the remaining DataPipes according to their relative weights.
    If you wish to maintain the same ratio of weights indefinitely, you need to ensure that the
    inputs are never exhausted, by, for instance, applying ``cycle`` to them.

    Sampling is controlled by the provided random ``seed``. If you don't provide it, the sampling
    will not be deterministic.

    Args:
        pipes_to_weights_dict: a `Dict` of IterDataPipes and Weights. The total weight of
            unexhausted DataPipes will be normalized to 1 for the purpose of sampling.
        seed: random seed to initialize the random number generator

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, SampleMultiplexer
        >>> source_dp1 = IterableWrapper([0] * 10)
        >>> source_dp2 = IterableWrapper([1] * 10)
        >>> d = {source_dp1: 99999999, source_dp2: 0.0000001}
        >>> sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d, seed=0)
        >>> list(sample_mul_dp)
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    """

    def __init__(
        self,
        pipes_to_weights_dict: Dict[IterDataPipe[T_co], float],
        seed: Optional[int] = None,
    ):
        if not pipes_to_weights_dict:
            raise ValueError("Empty dictionary passed to SampleMultiplexerDataPipe")
        total_weight: float = 0
        for v in pipes_to_weights_dict.values():
            if v <= 0:
                raise ValueError(f"Expecting a positive and non-zero weight, got {v}")
            total_weight += v

        self.pipes_and_weights = [(k, v / total_weight) for k, v in pipes_to_weights_dict.items()]
        if seed is None:
            self.random = random.Random()
        else:
            self.random = random.Random(seed)

    def __iter__(self) -> Iterator[T_co]:
        pipes_and_weights = [(iter(k), v) for k, v in self.pipes_and_weights]
        while len(pipes_and_weights) > 1:
            r = self.random.random()
            s: float = 0
            for it, weight in pipes_and_weights:
                s += weight
                if r < s:
                    try:
                        item = next(it)
                        yield item
                    except StopIteration:
                        # remove the current stream
                        new_total = 1 - weight
                        assert new_total > 0
                        pipes_and_weights = [(k, v / new_total) for k, v in pipes_and_weights if k != it]
                    break

        # only one stream left
        for item in pipes_and_weights[0][0]:
            yield item

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp, _ in self.pipes_and_weights):
            return sum(len(dp) for dp, _ in self.pipes_and_weights)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
