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
from torch.testing._internal.common_utils import IS_MACOS, TestCase
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


class TestStatefulDataLoaderIterable(TestCase):
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


class TestStatefulDataLoaderMap(TestStatefulDataLoaderIterable):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        if num_workers == 0:
            return
        dataset = DummyMapDataset(100, shuffle=shuffle)
        generator = torch.Generator()
        generator.manual_seed(13)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )

        if interrupt is None:
            interrupt = len(dl)

        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        exp = []
        for batch in it:
            exp.append(batch)

        # Restore new instance from state
        generator = torch.Generator()
        generator.manual_seed(13)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        batches = []
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


class TestStatefulSampler(TestStatefulDataLoaderIterable):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        dataset = DummyMapDataset(100, shuffle=shuffle)
        sampler = DummySampler(len(dataset))
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )

        if interrupt is None:
            interrupt = len(dl)

        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        exp = []
        for batch in it:
            exp.append(batch)

        # Restore new instance from state
        sampler = DummySampler(len(dataset))
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            batch_size=batch_size,
            sampler=sampler,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        batches = []
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


class GeneratorIterable(torch.utils.data.IterableDataset):
    def __init__(self, sizes_for_all_workers):
        self.sizes_for_all_workers = sizes_for_all_workers
        self.i = 0
        self.resume = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]
        self.i = 0
        if self.resume is not None:
            self.i = self.resume
            self.resume = None
        start = sum(self.sizes_for_all_workers[:worker_id])
        skip = self.i
        for i in range(start + skip, start + self.sizes_for_all_workers[worker_id]):
            self.i += 1
            yield i

    def state_dict(self):
        return {"i": self.i}

    def load_state_dict(self, state):
        self.resume = state["i"]


class GeneratorIterableNoState(torch.utils.data.IterableDataset):
    def __init__(self, sizes_for_all_workers):
        self.sizes_for_all_workers = sizes_for_all_workers

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]
        start = sum(self.sizes_for_all_workers[:worker_id])
        yield from range(start, start + self.sizes_for_all_workers[worker_id])


class TestStatefulDataLoaderGenerator(TestStatefulDataLoaderIterable):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        dataset = GeneratorIterable([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        exp = list(dl)

        if interrupt is None:
            interrupt = len(exp)

        dataset = GeneratorIterable([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        batches = []
        it = iter(dl)
        for _ in range(interrupt):
            batches.append(next(it))
        state_dict = dl.state_dict()

        self.assertEqual(batches, exp[:interrupt])

        # Restore new instance from state
        dataset = GeneratorIterable([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


class TestStatefulDataLoaderGeneratorNoState(TestStatefulDataLoaderIterable):
    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False):
        dataset = GeneratorIterableNoState([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        exp = list(dl)

        if interrupt is None:
            interrupt = len(exp)

        dataset = GeneratorIterableNoState([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        batches = []
        it = iter(dl)
        for _ in range(interrupt):
            batches.append(next(it))
        state_dict = dl.state_dict()

        self.assertEqual(batches, exp[:interrupt])

        # Restore new instance from state
        dataset = GeneratorIterableNoState([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        dl.load_state_dict(state_dict)
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


class TestSnapshotZero(TestCase):
    def test_generator(self):
        num_workers = 3
        every_n_steps = 10
        for pw in [False, True]:
            dataset = GeneratorIterable([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = list(it)

            dl.load_state_dict(state0)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_iterable(self):
        num_workers = 3
        every_n_steps = 10
        for pw in [False, True]:
            dataset = DummyIterableDataset([0, 100, 37], shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = list(it)

            dl.load_state_dict(state0)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map(self):
        num_workers = 3
        every_n_steps = 10
        for pw in [False, True]:
            dataset = DummyMapDataset(100, shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = list(it)

            dl.load_state_dict(state0)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map_shuffle(self):
        num_workers = 3
        every_n_steps = 10
        for pw in [False, True]:
            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                shuffle=True,  # Use default RandomSampler
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = list(it)

            dl.load_state_dict(state0)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map_iterrupted_shuffle(self):
        every_n_steps = 10

        for pw, num_workers, every_n_steps in itertools.product([False, True], [0, 2], [1, 15]):
            dataset = DummyMapDataset(10, shuffle=True)
            dl = StatefulDataLoader(
                dataset=dataset,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw if num_workers > 0 else False,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )

            it = iter(dl)
            state0 = dl.state_dict()
            exp = []
            for _ in range(4):
                exp.append(next(it))
            state1 = dl.state_dict()

            dl.load_state_dict(state1)
            it = iter(dl)
            for data in it:
                exp.append(data)

            dl.load_state_dict(state0)
            batches = []
            for data in iter(dl):
                batches.append(data)

            self.assertEqual(batches, exp)


class TestSnapshotEnd(TestCase):
    def test_generator(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = GeneratorIterable([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            exp = list(dl)
            state_end = dl.state_dict()

            batches = list(dl)  # simple restart
            self.assertEqual(batches, exp)

            dataset = GeneratorIterable([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            it = iter(dl)
            for _ in range(2):
                next(it)
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_generator_no_state(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = GeneratorIterableNoState([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            exp = list(dl)
            state_end = dl.state_dict()

            batches = list(dl)  # simple restart
            self.assertEqual(batches, exp)

            dataset = GeneratorIterableNoState([0, 100, 37])
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            it = iter(dl)
            for _ in range(2):
                next(it)
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_iterable(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = DummyIterableDataset([0, 100, 37], shuffle=True)
            g = torch.Generator()
            g.manual_seed(4)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                generator=g,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            list(dl)
            state_end = dl.state_dict()
            exp = list(dl)

            g.manual_seed(4)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                generator=g,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = DummyMapDataset(100, shuffle=True)
            generator = torch.Generator()
            generator.manual_seed(15)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                generator=generator,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            list(dl)
            state_end = dl.state_dict()
            exp = list(dl)

            generator.manual_seed(15)
            dl = StatefulDataLoader(
                dataset=dataset,
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                generator=generator,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)

    def test_map_shuffle(self):
        num_workers = 3
        every_n_steps = 10
        for pw, bs in itertools.product([False, True], [None, 4]):
            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                shuffle=True,  # Use default RandomSampler
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            list(dl)
            state_end = dl.state_dict()
            exp = list(dl)

            dataset = DummyMapDataset(100, shuffle=False)
            dl = StatefulDataLoader(
                dataset=dataset,
                shuffle=True,  # Use default RandomSampler
                num_workers=num_workers,
                collate_fn=identity,
                snapshot_every_n_steps=every_n_steps,
                persistent_workers=pw,
                batch_size=bs,
                multiprocessing_context=("forkserver" if IS_MACOS and num_workers else None),
            )
            dl.load_state_dict(state_end)
            batches = list(dl)

            self.assertEqual(batches, exp)


class TestNumWorkersMismatch(TestCase):
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


class TestTorchDataLazyImport(TestCase):
    def test_lazy_imports(self) -> None:
        import torchdata

        self.assertFalse("datapipes" in torchdata.__dict__)

        from torchdata import datapipes as dp, janitor  # noqa

        self.assertTrue("datapipes" in torchdata.__dict__)
        dp.iter.IterableWrapper([1, 2])


class TestConcurrentDataLoaders(TestCase):
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


class TestFastStateDictRequest(TestCase):
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


class TestJsonSerDe(TestCase):
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
