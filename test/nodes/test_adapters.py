# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase

from torch.utils.data import RandomSampler
from torchdata.nodes.adapters import IterableWrapper, MapStyleWrapper

from torchdata.nodes.types import Stateful

from .utils import DummyIterableDataset, DummyMapDataset, run_test_save_load_state


class _StatefulRange(Stateful):
    def __init__(self, n: int) -> None:
        self.n = n
        self._num_yielded = 0
        self._next_start = 0

    def __iter__(self) -> Iterator[int]:
        self._num_yielded = self._next_start  # Reset for next iter call
        self._next_start = 0
        for i in range(self._num_yielded, self.n):
            self._num_yielded += 1
            yield i

    def state_dict(self) -> Dict[str, Any]:
        return {"_num_yielded": self._num_yielded}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._next_start = state_dict["_num_yielded"]


class TestIterableWrapper(TestCase):
    def test_iterable(self):
        n = 20
        node = IterableWrapper(range(n))
        for epoch in range(2):
            result = list(node)
            self.assertEqual(len(result), n)
            for i, j in enumerate(result):
                self.assertEqual(j, i)

    def test_generator(self):
        n = 20
        node = IterableWrapper(f"str_{i}" for i in range(n))
        result = list(node)
        self.assertEqual(len(result), n)
        for i, j in enumerate(result):
            self.assertEqual(j, f"str_{i}")

        # Second time iter is called on generator will raise StopIteration
        result = list(node)
        self.assertEqual(len(result), 0)

    def test_iterable_dataset(self):
        n = 20
        node = IterableWrapper(DummyIterableDataset(n))
        for epoch in range(2):
            result = list(node)
            self.assertEqual(len(result), n)
            for i, row in enumerate(result):
                self.assertEqual(row["step"], i)
                self.assertEqual(row["test_tensor"].item(), i)
                self.assertEqual(row["test_str"], f"str_{i}")

    @parameterized.expand([0, 5])
    def test_save_load_state_fast_forward(self, midpoint: int):
        run_test_save_load_state(self, IterableWrapper(range(10)), midpoint)

    @parameterized.expand([0, 5])
    def test_save_load_state_stateful(self, midpoint: int):
        run_test_save_load_state(self, IterableWrapper(_StatefulRange(10)), midpoint)


class TestMapStyle(TestCase):
    def test_default_sampler(self):
        n = 20
        node = MapStyleWrapper(DummyMapDataset(n), sampler=range(n))
        for epoch in range(2):
            result = list(node)
            self.assertEqual(len(result), n)
            for i, row in enumerate(result):
                self.assertEqual(row["step"], i)
                self.assertEqual(row["test_tensor"].item(), i)
                self.assertEqual(row["test_str"], f"str_{i}")

    def test_random_sampler(self):
        n = 20
        ds = DummyMapDataset(n)
        node = MapStyleWrapper(ds, sampler=RandomSampler(ds))
        results = []
        for epoch in range(2):
            result = list(node)
            results.append(result)
            self.assertEqual(len(result), n)
            self.assertEqual({row["step"] for row in result}, set(range(n)))
            self.assertEqual({row["test_tensor"].item() for row in result}, set(range(n)))
            self.assertEqual(
                {row["test_str"] for row in result},
                {f"str_{i}" for i in range(n)},
            )

        self.assertNotEqual(results[0], results[1])  # Should have different values per epoch

    def test_dict(self):
        n = 20
        orig_ds = DummyMapDataset(n)
        d = {f"i{i}": orig_ds[i] for i in range(n)}
        sampler = list(d.keys())
        node = MapStyleWrapper(d, sampler=sampler)
        for epoch in range(2):
            result = list(node)
            self.assertEqual(len(result), n)
            for i, row in enumerate(result):
                self.assertEqual(row["step"], i)
                self.assertEqual(row["test_tensor"].item(), i)
                self.assertEqual(row["test_str"], f"str_{i}")

    @parameterized.expand([0, 7])
    def test_save_load_state_fast_forward(self, midpoint: int):
        n = 20
        node = MapStyleWrapper(DummyMapDataset(n), sampler=range(n))
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand([0, 7])
    def test_save_load_state_stateful(self, midpoint: int):
        n = 20
        node = MapStyleWrapper(DummyMapDataset(n), sampler=_StatefulRange(n))
        run_test_save_load_state(self, node, midpoint)
