# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, Optional, Sized

import torch.utils.data.sampler
from torch.utils.data.dataloader import _InfiniteConstantSampler

from .stateful import Stateful


class _StatefulRandomSamplerIterator(Iterator[int], Stateful):
    _GENERATOR = "generator"
    _YIELDED = "yielded"

    def __init__(self, sampler, parent_iterator: Iterator[int]):
        self.sampler = sampler
        self.parent_iterator = parent_iterator
        self.yielded = 0
        self.next_yielded = None
        self.generator_state = sampler.generator.get_state()

    def __next__(self) -> int:
        if self.next_yielded is not None:
            for _ in range(self.next_yielded):
                next(self.parent_iterator)

            self.yielded = self.next_yielded
            self.next_yielded = None

        val = next(self.parent_iterator)
        self.yielded += 1
        return val

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.generator_state = state_dict[self._GENERATOR]
        self.sampler.generator.set_state(state_dict[self._GENERATOR])
        self.next_yielded = state_dict[self._YIELDED]

    def state_dict(self) -> Dict[str, Any]:
        return {self._GENERATOR: self.generator_state, self._YIELDED: self.yielded}


class RandomSampler(torch.utils.data.sampler.RandomSampler):
    def __init__(
        self, data_source: Sized, replacement: bool = False, num_samples: Optional[int] = None, generator=None
    ):
        if generator is None:
            # Ensure that underlying sampler has something repeatable
            generator = torch.Generator()
            generator.manual_seed(1)
        super().__init__(data_source, replacement, num_samples, generator)

    def __iter__(self):
        return _StatefulRandomSamplerIterator(self, super().__iter__())


class BatchSampler(torch.utils.data.sampler.BatchSampler, Stateful):
    _SAMPLES_YIELDED = "samples_yielded"
    _SAMPLER_STATE = "sampler_state"
    _SAMPLER_ITER_STATE = "sampler_iter_state"

    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.samples_yielded = 0
        self.next_yielded = None
        self.sampler_iter = iter(sampler)

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {self._SAMPLES_YIELDED: self.samples_yielded}
        if isinstance(self.sampler, Stateful):
            sd[self._SAMPLER_STATE] = self.sampler.state_dict()
        if isinstance(self.sampler_iter, Stateful):
            sd[self._SAMPLER_ITER_STATE] = self.sampler_iter.state_dict()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.next_yielded = state_dict[self._SAMPLES_YIELDED]
        if self._SAMPLER_STATE in state_dict:
            assert isinstance(self.sampler, Stateful)
            self.sampler.load_state_dict(state_dict[self._SAMPLER_STATE])
        self.sampler_iter = iter(self.sampler)
        if self._SAMPLER_ITER_STATE in state_dict:
            assert isinstance(self.sampler_iter, Stateful)
            self.sampler_iter.load_state_dict(state_dict[self._SAMPLER_ITER_STATE])

    def __iter__(self):
        if self.next_yielded is not None:
            self.samples_yielded = self.next_yielded
            if not (isinstance(self.sampler, Stateful) or isinstance(self.sampler_iter, Stateful)) and not isinstance(
                self.sampler, _InfiniteConstantSampler
            ):
                # We skip x samples if underlying sampler is not stateful
                for _ in range(self.next_yielded):
                    next(self.sampler_iter)
            self.next_yielded = None
        elif self.samples_yielded > 0:
            # don't re-create sampler_iter unless necessary, we may already have one from init
            self.sampler_iter = iter(self.sampler)
            self.samples_yielded = 0

        if self.drop_last:
            while True:
                try:
                    batch = []
                    for _ in range(self.batch_size):
                        batch.append(next(self.sampler_iter))
                        self.samples_yielded += 1
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler_iter:
                self.samples_yielded += 1
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
