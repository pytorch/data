# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import unittest
from typing import Iterator

import torch
import torch.utils.data
from torch.testing._internal.common_utils import IS_MACOS
from torchdata.stateful_dataloader import Stateful, StatefulDataLoader


class DummyIterator(Iterator, Stateful):
    def __init__(self, samples, shuffle, include_generator):
        self.samples = samples
        self.shuffle = shuffle
        self.include_generator = include_generator
        self.size = len(self.samples)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration
        if self.shuffle:
            i = torch.randint(self.size, (1,)).item()
        else:
            i = self.i
        sample = self.samples[i]
        self.i += 1
        return sample

    def state_dict(self):
        sd = {"i": self.i}
        if self.include_generator:
            sd["g"] = torch.get_rng_state()
        return sd

    def load_state_dict(self, state_dict):
        self.i = state_dict["i"]
        if self.include_generator:
            torch.set_rng_state(state_dict["g"])


class DummySamplerIterator(Iterator, Stateful):
    def __init__(self, size):
        self.size = size
        self.i = 0

    def __next__(self):
        idx = self.i
        if idx >= self.size:
            raise StopIteration
        self.i += 1
        return idx

    def state_dict(self):
        return {"i": self.i}

    def load_state_dict(self, state_dict):
        self.i = state_dict["i"]


class DummySampler(torch.utils.data.Sampler):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        return DummySamplerIterator(self.size)

    def __len__(self):
        return self.size


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, sizes_for_all_workers, shuffle=False, include_generator=True):
        self.sizes_for_all_workers = sizes_for_all_workers
        self.shuffle = shuffle
        self.include_generator = include_generator

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]

        start = sum(self.sizes_for_all_workers[:worker_id])
        iter_data = list(range(start, start + self.sizes_for_all_workers[worker_id]))
        return DummyIterator(iter_data, self.shuffle, self.include_generator)


class DummyMapDataset(torch.utils.data.Dataset):
    def __init__(self, size, shuffle, include_generator=True):
        self.size = size
        self.data = [{"id": i, "strcol": f"strcol_{i}", "listcol": [i, i + 1, i + 2]} for i in range(size)]
        self.shuffle = shuffle
        self.include_generator = include_generator

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.shuffle:
            i = torch.randint(self.size, (1,)).item()
        return self.data[i]

    def state_dict(self):
        if self.include_generator:
            return {
                "g": torch.get_rng_state(),
            }
        else:
            return {}

    def load_state_dict(self, state_dict):
        if self.include_generator:
            torch.set_rng_state(state_dict["g"])


def identity(x):
    return x


class TestStatefulDataLoaderIterable(unittest.TestCase):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        dataset = DummyIterableDataset([0, 100, 37], shuffle=shuffle)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        list(dl)

        if interrupt is None:
            interrupt = len(exp)

        exp = []
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        for data in it:
            exp.append(data)

        # Restore new instance from state
        batches = []
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        for batch in iter(dl):
            batches.append(batch)

        self.assertEqual(exp, batches)

    def test_no_mp(self):
        for batch_size, interrupt in itertools.product([None, 7], [0, 1, 10]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=0,
                    batch_size=batch_size,
                    pw=False,
                    interrupt=interrupt,
                )

    def test_mp_x(self):
        for batch_size, interrupt in itertools.product([None, 7], [0, 1, 10]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=False,
                    interrupt=interrupt,
                )

    def test_mp_pw(self):
        for batch_size, interrupt in itertools.product([None, 7], [0, 1, 10]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=True,
                    interrupt=interrupt,
                )

    def test_mp_every_n_steps(self):
        batch_size = 7
        for every_n_steps, interrupt in itertools.product([2, 5], [0, 1, 10]):
            with self.subTest(every_n_steps=every_n_steps, batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=True,
                    interrupt=interrupt,
                )

    def test_random_state(self):
        for num_workers, interrupt in itertools.product([0, 3], [0, 1, 10]):
            with self.subTest(num_workers=num_workers, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=num_workers,
                    batch_size=7,
                    pw=False,
                    interrupt=interrupt,
                    shuffle=True,
                )


if __name__ == "__main__":
    unittest.main()
