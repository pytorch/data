# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from datasets.info import DatasetInfo
from datasets.iterable_dataset import ExamplesIterable, IterableDataset
from torch.testing._internal.common_utils import IS_MACOS, TestCase
from torchdata.stateful_dataloader import StatefulDataLoader


DEFAULT_N_EXAMPLES = 20
DEFAULT_FILEPATH = "file.txt"


def generate_examples_fn(**kwargs):
    kwargs = kwargs.copy()
    n = kwargs.pop("n", DEFAULT_N_EXAMPLES)
    filepaths = kwargs.pop("filepaths", None)
    for filepath in filepaths or [DEFAULT_FILEPATH]:
        if filepaths is not None:
            kwargs["filepath"] = filepath
        for i in range(n):
            yield f"{filepath}_{i}", {"id": i, **kwargs}


def identity(x):
    return x


class TestStatefulDataLoaderIterable_shard0(TestCase):
    def _get_dataset(self):
        ex_iterable = ExamplesIterable(generate_examples_fn, {})
        return IterableDataset(ex_iterable, info=DatasetInfo(description="dummy"), split="train")

    def _run_and_checkpoint(self, num_workers, batch_size, pw, interrupt, every_n_steps=1):
        dataset = self._get_dataset()
        dl = StatefulDataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=identity,
            snapshot_every_n_steps=every_n_steps,
            persistent_workers=pw,
            multiprocessing_context="forkserver" if IS_MACOS and num_workers else None,
        )
        it = iter(dl)
        for _ in range(interrupt):
            next(it)

        state_dict = dl.state_dict()
        exp = []
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
