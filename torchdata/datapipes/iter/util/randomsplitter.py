# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, final, List, Optional, TypeVar, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T = TypeVar("T")


@functional_datapipe("random_split")
class RandomSplitterIterDataPipe(IterDataPipe):
    r"""
    Randomly split samples from a source DataPipe into groups (functional name: ``random_split``).
    Since there is no buffer, only ONE group of samples (i.e. one child DataPipe) can be iterated through
    at any time. Attempts to iterate through multiple of them simultaneously will fail.

    Note that by default, multiple iterations of this DataPipe will yield the same split for consistency across epochs.
    You can invoke ``override_seed`` on the output(s) to update the seed whenever needed (such as per epoch to
    get a different split per epoch).

    Args:
        source_datapipe: Iterable DataPipe being split
        weights: Dict of weights; the length of this list determines how many output DataPipes there will be.
            It is recommended to provide integer weights that sum up to ``total_length``, which allows
            resulting DataPipes' length values to be known in advance.
        seed: random _seed used to determine the randomness of the split
        total_length: Length of the ``source_datapipe``, optional but providing an integer is highly encouraged,
            because not all ``IterDataPipe`` has ``len``, espeically ones that can be easily known in advance.
        target: Optional key (that must exist in ``weights``) to indicate the specific group to return.
            If set to the default ``None``, returns ``List[IterDataPipe]``.
            If target is specified, returns ``IterDataPipe``.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> train, valid = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0)
        >>> list(train)
        [2, 3, 5, 7, 8]
        >>> list(valid)
        [0, 1, 4, 6, 9]
        >>> # You can also specify a target key if you only need a specific group of samples
        >>> train = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0, target='train')
        >>> list(train)
        [2, 3, 5, 7, 8]
        >>> # Be careful to use the same seed as before when specifying `target` to get the correct split.
        >>> valid = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0, target='valid')
        >>> list(valid)
        [0, 1, 4, 6, 9]
    """

    def __new__(
        cls,
        source_datapipe: IterDataPipe,
        weights: Dict[T, Union[int, float]],
        seed,
        total_length: Optional[int] = None,
        target: Optional[T] = None,
    ):
        if total_length is None:
            try:
                # TODO: This is an issue for DataPipes which only have runtime lengths. Revisit to see if this
                #       is problematic.
                total_length = len(source_datapipe)
            except TypeError:
                raise TypeError(
                    "RandomSplitter needs `total_length`, but it is unable to infer it from "
                    f"the `source_datapipe`: {source_datapipe}."
                )
        container = _RandomSplitterIterDataPipe(source_datapipe, total_length, weights, seed)
        if target is None:
            return [SplitterIterator(container, k) for k in list(weights.keys())]
        else:
            if target in weights.keys():
                return SplitterIterator(container, target)
            else:
                raise KeyError(f"`target={target}` does not match any key in `weights`.")


class _RandomSplitterIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe,
        total_length: int,
        weights: Dict[T, Union[int, float]],
        seed,
    ):
        self.source_datapipe: IterDataPipe = source_datapipe
        self.total_length: int = total_length
        self.remaining_length: int = total_length
        self._seed = seed
        self.keys: List[T] = list(weights.keys())
        self.key_to_index: Dict[T, int] = {k: i for i, k in enumerate(self.keys)}
        self.norm_weights: List[float] = self.normalize_weights([weights[k] for k in self.keys], total_length)
        self.weights: List[float] = self.norm_weights.copy()
        self._rng = random.Random(self._seed)
        self._lengths: List[int] = []

    def draw(self) -> T:
        selected_key = self._rng.choices(self.keys, self.weights)[0]
        index = self.key_to_index[selected_key]
        self.weights[index] -= 1
        self.remaining_length -= 1
        if self.weights[index] < 0:
            self.weights[index] = 0
            self.weights = self.normalize_weights(self.weights, self.remaining_length)
        return selected_key

    @staticmethod
    def normalize_weights(weights: List[float], total_length: int) -> List[float]:
        """
        Given a ``List`` of weights, normalize them according to ``total_length``.
        """
        total_weight = sum(weights)
        return [float(w) * total_length / total_weight for w in weights]

    @final
    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self.weights = self.norm_weights.copy()
        self.remaining_length = self.total_length

    def override_seed(self, seed):
        """
        Update the `seed`. The new `seed` will be used in the next iteration.
        """
        self._seed = seed
        return self

    def __getstate__(self):
        state = (
            self.source_datapipe,
            self.total_length,
            self._seed,
            self.norm_weights,
            self.keys,
            self.key_to_index,
            self.weights,
            self._rng.getstate(),
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.source_datapipe,
            self.total_length,
            self._seed,
            self.norm_weights,
            self.keys,
            self.key_to_index,
            self.weights,
            rng_state,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)

    def get_length(self, target: T) -> int:
        if not self._lengths:
            if all(w.is_integer() for w in self.norm_weights) and sum(self.norm_weights) == self.total_length:
                self._lengths = [int(w) for w in self.norm_weights]
            else:
                raise TypeError(
                    "Lengths of the split cannot be known in advance. Please supply "
                    "integer `weights` that sum up to `total_length`.\nAlternatively, "
                    "use `datapipe.set_length(LENGTH)` to manually set the desired length."
                )
        index = self.key_to_index[target]
        return self._lengths[index]


class SplitterIterator(IterDataPipe):
    def __init__(self, main_datapipe: _RandomSplitterIterDataPipe, target: T):
        self.main_datapipe = main_datapipe
        self.target = target

    def __iter__(self):
        self.main_datapipe.reset()
        for sample in self.main_datapipe.source_datapipe:
            if self.main_datapipe.draw() == self.target:
                yield sample

    def override_seed(self, seed):
        """
        Update the `seed`. The new `seed` will be used in the next iteration. For use cases that require a different
        split for each epoch, you call this method before or after the epoch as necessary.
        """
        self.main_datapipe.override_seed(seed)
        return self

    def __len__(self):
        return self.main_datapipe.get_length(self.target)
