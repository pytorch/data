# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from typing import List, Literal

import testslide

from parameterized import parameterized
from torch.testing._internal.common_utils import IS_WINDOWS, TEST_CUDA
from torchdata.nodes.batch import Batcher
from torchdata.nodes.map import Mapper, ParallelMapper
from torchdata.nodes.pin_memory import PinMemory
from torchdata.nodes.prefetch import Prefetcher

from .utils import MockSource, RandomSleepUdf, run_test_save_load_state, udf_raises


class TestMap(unittest.TestCase):
    def _test_exception_handling_mapper(self, pin_memory, method):
        batch_size = 6
        multiprocessing_context = None if IS_WINDOWS else "forkserver"
        src = MockSource(num_samples=20)
        node = Batcher(src, batch_size=batch_size)
        node = ParallelMapper(
            node,
            udf_raises,
            num_workers=2,
            method=method,
            multiprocessing_context=multiprocessing_context,
        )
        node = Mapper(node, udf_raises)
        if pin_memory:
            node = PinMemory(node)
        node = Prefetcher(node, prefetch_factor=2)

        with self.assertRaisesRegex(ValueError, "test exception"):
            list(node)

    def test_exception_handling_mapper(self):
        self._test_exception_handling_mapper(False, "thread")

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_exception_handling_mapper_cuda(self):
        self._test_exception_handling_mapper(True, "thread")

    def test_exception_handling_mapper_multiprocess(self):
        self._test_exception_handling_mapper(False, "process")

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_exception_handling_mapper_multiprocess_cuda(self):
        self._test_exception_handling_mapper(True, "process")

    def _test_map(self, in_order, method) -> None:
        batch_size = 6
        n = 80
        multiprocessing_context = None if IS_WINDOWS else "forkserver"
        src = MockSource(num_samples=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = ParallelMapper(
            node,
            RandomSleepUdf(),
            num_workers=4,
            in_order=in_order,
            method=method,
            multiprocessing_context=multiprocessing_context,
        )
        node = Prefetcher(node, prefetch_factor=2)

        results: List[List[dict]] = [[], []]
        for epoch in range(2):
            for batch in node:
                results[epoch].extend(batch)

        for result in results:
            self.assertEqual(len(result), n, epoch)
            if in_order:
                for i, row in enumerate(result):
                    self.assertEqual(row["step"], i, epoch)
                    self.assertEqual(row["test_tensor"].item(), i, epoch)
                    self.assertEqual(row["test_str"], f"str_{i}", epoch)
            else:
                self.assertEqual({row["step"] for row in result}, set(range(n))), epoch
                self.assertEqual(
                    {row["test_tensor"].item() for row in result},
                    set(range(n)),
                    epoch,
                )
                self.assertEqual(
                    {row["test_str"] for row in result},
                    {f"str_{i}" for i in range(n)},
                    epoch,
                )

    def test_in_order_threads(self):
        self._test_map(True, "thread")

    def test_out_of_order_threads(self):
        self._test_map(False, "thread")

    def test_in_order_process(self):
        self._test_map(True, "process")

    def test_out_of_order_process(self):
        self._test_map(False, "process")

    @parameterized.expand(
        itertools.product(
            [0, 7, 14],
            [True],  # TODO: define and fix in_order = False
            [0, 1, 9],  # TODO: define and fix in_order = False
        )
    )
    def test_save_load_state_thread(
        self, midpoint: int, in_order: bool, snapshot_frequency: int
    ):
        method = "thread"
        batch_size = 6
        n = 80
        multiprocessing_context = None if IS_WINDOWS else "forkserver"
        src = MockSource(num_samples=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = ParallelMapper(
            node,
            RandomSleepUdf(),
            num_workers=4,
            in_order=in_order,
            method=method,
            multiprocessing_context=multiprocessing_context,
            snapshot_frequency=snapshot_frequency,
        )
        node = Prefetcher(node, prefetch_factor=2)
        run_test_save_load_state(self, node, midpoint)

    @parameterized.expand(
        itertools.product(
            [0, 7, 14],
            [True],  # TODO: define and fix in_order = False
            [0, 1, 9],  # TODO: define and fix in_order = False
        )
    )
    def test_save_load_state_process(
        self, midpoint: int, in_order: bool, snapshot_frequency: int
    ):
        method = "process"
        batch_size = 6
        n = 80
        multiprocessing_context = None if IS_WINDOWS else "forkserver"
        src = MockSource(num_samples=n)
        node = Batcher(src, batch_size=batch_size, drop_last=False)
        node = ParallelMapper(
            node,
            RandomSleepUdf(),
            num_workers=4,
            in_order=in_order,
            method=method,
            multiprocessing_context=multiprocessing_context,
            snapshot_frequency=snapshot_frequency,
        )
        node = Prefetcher(node, prefetch_factor=2)
        run_test_save_load_state(self, node, midpoint)
