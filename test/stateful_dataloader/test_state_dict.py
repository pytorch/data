import copy
import itertools
import unittest
from typing import Iterator

import torch

from torchdata.stateful_dataloader import Stateful, StatefulDataLoader


class DummyIterator(Iterator, Stateful):
    def __init__(self, samples, shuffle):
        self.samples = samples
        self.shuffle = shuffle
        self.g = torch.Generator()
        self.g.manual_seed(1)
        self.size = len(self.samples)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration
        if self.shuffle:
            i = torch.randint(self.size, (1,), generator=self.g).item()
        else:
            i = self.i
        sample = self.samples[i]
        self.i += 1
        return sample

    def state_dict(self):
        return {"i": self.i, "g": self.g.get_state()}

    def load_state_dict(self, state_dict):
        self.i = state_dict["i"]
        self.g.set_state(state_dict["g"])


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, sizes_for_all_workers, shuffle=False):
        self.sizes_for_all_workers = sizes_for_all_workers
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]

        start = sum(self.sizes_for_all_workers[:worker_id])
        iter_data = list(range(start, start+self.sizes_for_all_workers[worker_id]))
        return DummyIterator(iter_data, self.shuffle)


class DummyMapDataset(torch.utils.data.Dataset):
    def __init__(self, size, shuffle):
        self.size = size
        self.data = [
            {"id": i, "strcol": f"strcol_{i}", "listcol": [i, i + 1, i + 2]}
            for i in range(size)
        ]
        self.shuffle = shuffle
        self.g = torch.Generator()
        self.g.manual_seed(1)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.shuffle:
            i = torch.randint(self.size, (1,), generator=self.g).item()
        return self.data[i]

    def state_dict(self):
        return {
            "g": self.g.get_state(),
        }

    def load_state_dict(self, state_dict):
        self.g.set_state(state_dict["g"])


def identity(x):
    return x


class TestStatefulDataLoaderIterable(unittest.TestCase):
    def _run_and_checkpoint(
        self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False
    ):
        dataset = DummyIterableDataset([0, 100, 37], shuffle=shuffle)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
        )
        exp = list(dl)

        if interrupt == None:
            interrupt = len(exp)

        batches = []
        it = iter(dl)
        for i in range(interrupt):
            batches.append(next(it))
        state_dict = dl.state_dict()

        self.assertEqual(batches, exp[:interrupt])

        # Restore new instance from state
        dataset = DummyIterableDataset([0, 100, 37], shuffle=shuffle)
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
        )
        dl.load_state_dict(state_dict)
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)

    def test_no_mp(self):
        for batch_size, interrupt in itertools.product([None, 7], [1, 10, None]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=0,
                    batch_size=batch_size,
                    pw=False,
                    interrupt=interrupt,
                )

    def test_mp_x(self):
        for batch_size, interrupt in itertools.product([None, 7], [1, 10, None]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=False,
                    interrupt=interrupt,
                )

    def test_mp_pw(self):
        for batch_size, interrupt in itertools.product([None, 7], [1, 10, None]):
            with self.subTest(batch_size=batch_size, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=True,
                    interrupt=interrupt,
                )

    def test_mp_every_n_steps(self):
        batch_size = 7
        for every_n_steps, interrupt in itertools.product([2, 5], [1, 10, None]):
            with self.subTest(
                every_n_steps=every_n_steps, batch_size=batch_size, interrupt=interrupt
            ):
                self._run_and_checkpoint(
                    num_workers=3,
                    batch_size=batch_size,
                    pw=True,
                    interrupt=interrupt,
                )

    def test_random_state(self):
        for num_workers, interrupt in itertools.product([0, 3], [1, 10, None]):
            with self.subTest(num_workers=num_workers, interrupt=interrupt):
                self._run_and_checkpoint(
                    num_workers=num_workers,
                    batch_size=7,
                    pw=False,
                    interrupt=interrupt,
                    shuffle=True,
                )

class TestStatefulDataLoaderMap(TestStatefulDataLoaderIterable):
    def _run_and_checkpoint(
        self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False
    ):
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
        )

        if interrupt == None:
            interrupt = len(dl)

        it = iter(dl)
        for i in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        exp = []
        for batch in it:
            exp.append(batch)

        # Restore new instance from state
        # dataset = DummyMapDataset(100, shuffle=shuffle)
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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
            self.sizes_for_all_workers = [sum(self.sizes_for_all_workers)]

        start = sum(self.sizes_for_all_workers[:worker_id])
        skip = self.i
        for i in range(start+skip, start+self.sizes_for_all_workers[worker_id]):
            self.i += 1
            yield i

    def state_dict(self):
        return {"i": self.i}

    def load_state_dict(self, state):
        self.i = state["i"]


class TestStatefulDataLoaderGenerator(TestStatefulDataLoaderIterable):
    def _run_and_checkpoint(
        self, num_workers, batch_size, pw, interrupt, every_n_steps=1, shuffle=False
    ):
        dataset = GeneratorIterable([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
        )
        exp = list(dl)

        if interrupt == None:
            interrupt = len(exp)

        dataset = GeneratorIterable([0, 100, 37])
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
        )
        batches = []
        it = iter(dl)
        for i in range(interrupt):
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
        )
        dl.load_state_dict(state_dict)
        for batch in dl:
            batches.append(batch)

        self.assertEqual(batches, exp)


if __name__ == "__main__":
    unittest.main()
