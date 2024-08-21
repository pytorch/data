# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, Optional, Sized

import torch.utils.data.sampler
from torch.utils.data.dataloader import _InfiniteConstantSampler
from torch.utils.data import Dataset, Sampler

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


class _StatefulDistributedSamplerIterator(Iterator[int], Stateful):

    def __init__(self, sampler):
        self.sampler = sampler

        if self.sampler.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.sampler.seed + self.sampler.epoch)
            indices = torch.randperm(len(self.sampler.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.sampler.dataset)))  # type: ignore[arg-type]

        if not self.sampler.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.sampler.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.sampler.total_size]
        assert len(indices) == self.sampler.total_size

        # subsample
        indices = indices[self.sampler.rank : self.sampler.total_size : self.sampler.num_replicas]
        assert len(indices) == self.sampler.num_samples

        self.parent_iterator = iter(indices)
        self.indices = list(self.parent_iterator)
        self.current_index = 0

    def __next__(self) -> int:
        if self.sampler.next_yielded is not None:
            self.current_index = self.sampler.next_yielded
            self.sampler.yielded = self.sampler.next_yielded
            self.sampler.next_yielded = None
        if self.current_index >= len(self.indices):
            raise StopIteration  

        val = self.indices[self.current_index]
        self.current_index += 1
        self.sampler.yielded += 1
        return val


class StatefulDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    _YIELDED = "yielded"

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:

        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.yielded = 0
        self.next_yielded = None

    def __iter__(self):

        return _StatefulDistributedSamplerIterator(self)

    def state_dict(self) -> Dict[str, Any]:
        return {self._YIELDED: self.yielded}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict[self._YIELDED] < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")

        self.next_yielded = state_dict[self._YIELDED]
