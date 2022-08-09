# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from copy import deepcopy
from functools import partial
from typing import List, Sequence, TypeVar, Union

from torch.utils.data.datapipes.iter.combining import _ChildDataPipe, _DemultiplexerIterDataPipe
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

K = TypeVar("K")


@functional_datapipe("random_split")
class RandomSplitterIterDataPipe(IterDataPipe):
    r"""
    Randomly split samples from a source DataPipe into multiple DataPipes (functional name: ``random_split``).
    Samples are temporily stored in a buffer such that child DataPipes can be used simultaneously.
    Note that multiple iterations of this DataPipe will yield the same split for consistency across epochs.

    Args:
        source_datapipe: Iterable DataPipe being split
        total_length: Length of the source
        weights: a list of weights; the length of this list determines how many output DataPipes there will be.
        seed: random seed used to determine the split
        buffer_size: determines the maximum number of unread

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> train, validation = dp.random_split(total_length=10, weights=[0.5, 0.5], seed=0)
        >>> list(train)
        [1, 3, 4, 5, 9]
    """

    def __new__(
        cls,
        source_datapipe: IterDataPipe[K],
        total_length: int,
        weights: Sequence[Union[int, float]],
        seed,
        buffer_size: int = 1000,
    ):
        norm_weights = cls.normalize_weights(weights, total_length)
        instance_ids = list(range(len(weights)))

        # The implementation basically uses Demux but with a custom classifier function and additional reset operation
        container = _RandomSplitterIterDataPipe(
            source_datapipe, instance_ids, seed, norm_weights, buffer_size
        )  # type: ignore[arg-type]
        return [_ChildDataPipe(container, i) for i in range(len(instance_ids))]

    @staticmethod
    def normalize_weights(weights, total_length):
        total_weight = sum(weights)
        return [round(float(w) * total_length / total_weight) for w in weights]


class _RandomSplitterIterDataPipe(_DemultiplexerIterDataPipe):
    def __init__(
        self,
        datapipe: IterDataPipe,
        instance_ids: List[int],
        seed,
        initial_weights: Sequence[Union[int, float]],
        buffer_size: int = 1000,
    ):
        super().__init__(
            datapipe, len(instance_ids), classifier_fn=None, drop_none=False, buffer_size=buffer_size
        )  # type: ignore[arg-type]
        self.instance_ids = instance_ids
        self.seed = seed
        self.initial_weights = initial_weights

        self.rng = random.Random(self.seed)
        self.classifier_fn = partial(self.draw, instance_ids, deepcopy(self.initial_weights), self.rng)

    def reset(self):
        super().reset()
        self.rng = random.Random(self.seed)
        self.classifier_fn = partial(self.draw, self.instance_ids, deepcopy(self.initial_weights), self.rng)

    @staticmethod
    def draw(population, weights, rng, _):
        idx = rng.choices(population, weights)[0]
        weights[idx] -= 1
        return idx
