# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, TypeVar, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T = TypeVar("K")


@functional_datapipe("random_split")
class RandomSplitterIterDataPipe(IterDataPipe):
    r"""
    Randomly split samples from a source DataPipe into groups(functional name: ``random_split``).
    Since there is no buffer, only ONE group of samples (i.e. one child DataPipe) can be iterated through
    at any time. Attempts to iterate through multiple of them simultaneously will fail.
    Note that multiple iterations of this DataPipe will yield the same split for consistency across epochs.

    Args:
        source_datapipe: Iterable DataPipe being split
        total_length: Length of the source
        weights: Dict of weights; the length of this list determines how many output DataPipes there will be.
        seed: random seed used to determine the split

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> train, valid = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0)
        >>> list(train)
        [2, 3, 5, 7, 8]
        >>> list(valid)
        [0, 1, 4, 6, 9]
    """

    def __new__(
        cls,
        source_datapipe: IterDataPipe,
        total_length: int,
        weights: Dict[T, Union[int, float]],
        seed,
    ):
        container = _RandomSplitterIterDataPipe(source_datapipe, total_length, weights, seed)
        return [SplitterIterator(container, k) for k in list(weights.keys())]


class _RandomSplitterIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        total_length: int,
        weights: Dict[T, Union[int, float]],
        seed,
    ):
        self.source_datapipe = source_datapipe
        self.total_length = total_length
        self.seed = seed
        self.norm_weights = self.normalize_weights(weights, total_length)
        self.keys = list(weights.keys())
        self.key_to_index = {k: i for i, k in enumerate(self.keys)}
        self.weights = [self.norm_weights[k] for k in self.keys]
        self.rng = random.Random(self.seed)

    def draw(self):
        selected_key = self.rng.choices(self.keys, self.weights)[0]
        self.weights[self.key_to_index[selected_key]] -= 1
        return selected_key

    @staticmethod
    def normalize_weights(weights, total_length: int):
        total_weight = sum(weights.values())
        return {k: round(float(w) * total_length / total_weight) for k, w in weights.items()}

    def reset(self):
        self.rng = random.Random(self.seed)
        self.weights = [self.norm_weights[k] for k in self.keys]


class SplitterIterator:
    def __init__(self, main_datapipe: _RandomSplitterIterDataPipe, target: T):
        self.main_datapipe = main_datapipe
        self.target = target

    def __iter__(self):
        self.main_datapipe.reset()
        for sample in self.main_datapipe.source_datapipe:
            if self.main_datapipe.draw() == self.target:
                yield sample
