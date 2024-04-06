from typing import Any, Dict, Optional, Sized

import torch.utils.data.sampler
from torch.utils.data.dataloader import _InfiniteConstantSampler

from .stateful import Stateful


class RandomSampler(torch.utils.data.sampler.RandomSampler, Stateful):
    def __init__(
        self, data_source: Sized, replacement: bool = False, num_samples: Optional[int] = None, generator=None
    ):

        if generator is None:
            # Ensure that underlying sampler has something repeatable
            generator = torch.Generator()
            generator.manual_seed(1)
        super().__init__(data_source, replacement, num_samples, generator)
        self.yielded = 0
        self.next_yielded = None

    def state_dict(self) -> Dict[str, Any]:
        return {"generator": self.generator.get_state() if self.generator else None, "yielded": self.yielded}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict["generator"] is not None:
            self.generator.set_state(state_dict["generator"])
        self.next_yielded = state_dict["yielded"]

    def __iter__(self):
        super_iter = super().__iter__()
        self.yielded = self.next_yielded or 0
        while True:
            try:
                val = next(super_iter)
                yield val
                self.yielded += 1
            except StopIteration:
                return


torch.utils.data.sampler.RandomSampler = RandomSampler  # type: ignore[misc]
torch.utils.data.dataloader.RandomSampler = RandomSampler  # type: ignore[misc]


class BatchSampler(torch.utils.data.sampler.BatchSampler, Stateful):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.samples_yielded = 0
        self.next_yielded = None
        self.sampler_iter = iter(sampler)

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {"samples_yielded": self.samples_yielded}
        if isinstance(self.sampler, Stateful):
            sd["sampler"] = self.sampler.state_dict()
        if isinstance(self.sampler_iter, Stateful):
            sd["sampler_iter"] = self.sampler_iter.state_dict()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.next_yielded = state_dict["samples_yielded"]
        if "sampler" in state_dict:
            assert isinstance(self.sampler, Stateful)
            self.sampler.load_state_dict(state_dict["sampler"])
        self.sampler_iter = iter(self.sampler)
        if "sampler_iter" in state_dict:
            assert isinstance(self.sampler_iter, Stateful)
            self.sampler_iter.load_state_dict(state_dict["sampler_iter"])

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


torch.utils.data.sampler.BatchSampler = BatchSampler  # type: ignore[misc]
torch.utils.data.dataloader.BatchSampler = BatchSampler  # type: ignore[misc]
