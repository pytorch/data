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


class TestNumWorkersMismatch(unittest.TestCase):
    def test_num_workers_mismatch(self):
        for initial_num_workers, num_workers in ((0, 3), (3, 0)):
            if initial_num_workers == num_workers:
                continue
            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=initial_num_workers,
                collate_fn=identity,
                multiprocessing_context=("forkserver" if IS_MACOS and initial_num_workers else None),
            )
            state = dl.state_dict()
            self.assertEqual(len(state), 0)

            iter(dl)
            state = dl.state_dict()
            self.assertTrue(len(state) > 0)

            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            dl.load_state_dict(state)
            try:
                iter(dl)
                raise Exception("Expected AssertionError to be thrown")
            except AssertionError:
                continue
            self.assertTrue(False, "Error should be of type AssertionError")


class TestTorchDataLazyImport(unittest.TestCase):
    def test_lazy_imports(self) -> None:
        import torchdata

        self.assertFalse("datapipes" in torchdata.__dict__)

        from torchdata import datapipes as dp, janitor  # noqa

        self.assertTrue("datapipes" in torchdata.__dict__)
        dp.iter.IterableWrapper([1, 2])


class TestConcurrentDataLoaders(unittest.TestCase):
    def test_two_dataloaders(self) -> None:
        dataset = DummyMapDataset(100, shuffle=False)
        sdl = StatefulDataLoader(
            dataset=dataset,
            num_workers=2,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS else None,
        )
        exp = list(sdl)

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=2,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS else None,
        )
        data = list(dl)
        self.assertEqual(data, exp)


class TestFastStateDictRequest(unittest.TestCase):
    def _run_test(self, snapshot_every_n_steps, interrupt):
        num_workers = 4
        dataset = DummyIterableDataset([25, 25, 25, 25], shuffle=True)

        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=4,
            collate_fn=identity,
            persistent_workers=True,
            multiprocessing_context="forkserver" if IS_MACOS else None,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        for _ in range(2):
            next(it)
        exp = list(it)

        dl.load_state_dict(state_dict)
        # new iter after load_state_dict, ask for state dict before num_workers batches
        # are yielded to ensure old worker states are stored properly
        it = iter(dl)
        for _ in range(2):
            next(it)

        state_dict2 = dl.state_dict()
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=4,
            collate_fn=identity,
            persistent_workers=True,
            multiprocessing_context="forkserver" if IS_MACOS else None,
        )
        dl.load_state_dict(state_dict2)
        data = list(dl)

        self.assertEqual(data, exp)

    def test_fast_state_dict_request(self) -> None:
        self._run_test(0, 11)

    def test_fast_state_dict_request_skip_steps(self) -> None:
        self._run_test(17, 19)


class TestJsonSerDe(unittest.TestCase):
    def _run_test_iterable(self, num_workers):
        interrupt = 4
        dataset = DummyIterableDataset([0, 100, 37], shuffle=False, include_generator=False)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        list(dl)

        exp = []
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        ser = json.dumps(state_dict)
        for data in it:
            exp.append(data)

        # Restore new instance from state
        batches = []
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        deser = json.loads(ser)
        dl.load_state_dict(deser)
        for batch in iter(dl):
            batches.append(batch)

        self.assertEqual(exp, batches)

    def _run_test_map(self, num_workers):
        interrupt = 4
        dataset = DummyMapDataset(100, shuffle=False, include_generator=False)
        sampler = DummySampler(100)
        dl = StatefulDataLoader(
            dataset=dataset,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        list(dl)

        exp = []
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        ser = json.dumps(state_dict)
        for data in it:
            exp.append(data)

        # Restore new instance from state
        batches = []
        dl = StatefulDataLoader(
            dataset=dataset,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=identity,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        deser = json.loads(ser)
        dl.load_state_dict(deser)
        for batch in iter(dl):
            batches.append(batch)

        self.assertEqual(exp, batches)

    def test_json_serde_single_process(self):
        self._run_test_iterable(0)

    def test_json_serde_multi_process(self):
        self._run_test_iterable(3)

    def test_json_serde_single_process_map(self):
        self._run_test_map(0)

    def test_json_serde_multi_process_map(self):
        self._run_test_map(3)


if __name__ == "__main__":
    unittest.main()
