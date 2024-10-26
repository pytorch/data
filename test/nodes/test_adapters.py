# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import testslide
from torch.utils.data import IterableDataset, RandomSampler
from torchdata.nodes.adapters import IterableWrapper, MapStyleWrapper, ToIterableDataset

from .utils import DummyIterableDataset, DummyMapDataset, MockSource


class TestIterableWrapper(testslide.TestCase):
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


class TestMapStyle(testslide.TestCase):
    def test_default_sampler(self):
        n = 20
        node = MapStyleWrapper(DummyMapDataset(n))
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


class TestToIterableDataset(testslide.TestCase):
    def test_to_iterable_dataset(self):
        n = 20
        node = MockSource(n)
        iterable_ds = ToIterableDataset(node)
        self.assertIsInstance(iterable_ds, IterableDataset)
        for epoch in range(2):
            result = list(iterable_ds)
            self.assertEqual(len(result), n)
            for i, j in enumerate(result):
                self.assertEqual(j, i)
