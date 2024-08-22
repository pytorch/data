# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import errno
import faulthandler
import functools
import gc
import itertools
import math
import operator
import os
import signal
import sys
import tempfile
import time
import unittest
import warnings

import torch
import torch.utils.data.datapipes as dp
from torch import multiprocessing as mp
from torch._utils import ExceptionWrapper
from torch.testing._internal.common_device_type import instantiate_device_type_tests

from torch.testing._internal.common_utils import (
    IS_CI,
    IS_JETSON,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
    NO_MULTIPROCESSING_SPAWN,
    parametrize,
    run_tests,
    skipIfNoDill,
    skipIfRocm,
    slowTest,
    TEST_CUDA,
    TEST_NUMPY,
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
    TestCase,
)

from torch.utils.data import (
    _utils,
    ChainDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
    IterDataPipe,
    StackDataset,
    Subset,
    TensorDataset,
)
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.dataset import random_split

from torchdata.stateful_dataloader import Stateful, StatefulDataLoader, StatefulDataLoader as DataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    err_msg = (
        "psutil not found. Some critical data loader tests relying on it "
        "(e.g., TestDataLoader.test_proper_exit) will not run."
    )
    if IS_CI:
        raise ImportError(err_msg) from None
    else:
        warnings.warn(err_msg)


try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
skipIfNoNumpy = unittest.skipIf(not HAS_NUMPY, "no NumPy")

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if TEST_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

if not NO_MULTIPROCESSING_SPAWN:
    # We want to use `spawn` if able because some of our tests check that the
    # data loader terminiates gracefully. To prevent hanging in the testing
    # process, such data loaders are run in a separate subprocess.
    #
    # We also want to test the `pin_memory=True` configuration, thus `spawn` is
    # required to launch such processes and they initialize the CUDA context.
    #
    # Mixing different start method is a recipe for disaster (e.g., using a fork
    # `mp.Event` with a spawn `mp.Process` segfaults). So we set this globally
    # to avoid bugs.
    #
    # Get a multiprocessing context because some test / third party library will
    # set start_method when imported, and setting again triggers `RuntimeError`.
    mp = mp.get_context(method="spawn")


# 60s of timeout?
# Yes, in environments where physical CPU resources are shared, e.g., CI, the
# time for a inter-process communication can be highly varying.  With 15~17s of
# timeout, we have observed flakiness in some CI builds (see
# pytorch/pytorch#14501, pytorch/pytorch#16608).  We follow the CPython
# multiprocessing setup and set the timeout to 60s here:
#
# https://github.com/python/cpython/blob/e8113f51a8bdf33188ee30a1c038a298329e7bfa/Lib/test/_test_multiprocessing.py#L73
JOIN_TIMEOUT = 60.0  # seconds


supported_multiprocessing_contexts = [None] + list(torch.multiprocessing.get_all_start_methods())


# collate_fn that returns the batch cloned; defined globally here for pickle purposes.
def _clone_collate(b):
    return [x.clone() for x in b]


class MockDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.arange(size)  # Simple data that is easy to verify

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
@unittest.skipIf(TEST_WITH_ASAN, "DataLoader tests hang in ASAN, see: https://github.com/pytorch/pytorch/issues/66223")
class TestDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = MockDataset(100)
        self.persistent_workers = False

    def test_initialization_StatefulDistributedSampler(self):

        sampler = StatefulDistributedSampler(
            self.dataset, num_replicas=10, rank=0, shuffle=False, seed=42, drop_last=False
        )
        self.assertEqual(sampler.dataset, self.dataset)
        self.assertEqual(sampler.num_replicas, 10)
        self.assertEqual(sampler.rank, 0)
        self.assertFalse(sampler.shuffle)
        self.assertEqual(sampler.seed, 42)
        self.assertFalse(sampler.drop_last)
        self.assertEqual(sampler.yielded, 0)
        self.assertIsNone(sampler.next_yielded)

    def test_dataloader_state_dict(self):
        sampler = StatefulDistributedSampler(self.dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = StatefulDataLoader(self.dataset, batch_size=10, sampler=sampler)
        # Partial iteration over the DataLoader
        iter_count = 5
        for i, data in enumerate(dataloader):
            if i == iter_count - 1:
                break
        state_dict = dataloader.state_dict()
        new_sampler = StatefulDistributedSampler(self.dataset, num_replicas=1, rank=0, shuffle=False)

        new_dataloader = StatefulDataLoader(self.dataset, batch_size=10, sampler=new_sampler)
        new_dataloader.load_state_dict(state_dict)
        resumed_data = []
        for data in new_dataloader:
            resumed_data.append(data.tolist())
        expected_data = []
        full_dataloader = DataLoader(self.dataset, batch_size=10, sampler=sampler)
        for data in full_dataloader:
            expected_data.append(data.tolist())

        self.assertEqual(resumed_data, expected_data[iter_count:])

    def test_sampler_state_dict(self):

        sampler = StatefulDistributedSampler(self.dataset, num_replicas=10, rank=0)
        sampler.yielded = 5
        state_dict = sampler.state_dict()
        self.assertEqual(state_dict["yielded"], 5)

    def test_sampler_load_state_dict(self):

        sampler = StatefulDistributedSampler(self.dataset, num_replicas=10, rank=0)
        sampler.load_state_dict({"yielded": 3})
        self.assertEqual(sampler.next_yielded, 3)
        with self.assertRaises(ValueError):
            sampler.load_state_dict({"yielded": -1})

    def test_sampler_next_yielded(self):

        sampler = StatefulDistributedSampler(self.dataset, num_replicas=2, rank=0, shuffle=True, seed=42)
        iterator = iter(sampler)
        next(iterator)  # advance the iterator
        self.assertEqual(sampler.yielded, 1)
        self.assertIsNone(sampler.next_yielded)
        sampler.load_state_dict({StatefulDistributedSampler._YIELDED: 5})
        self.assertEqual(sampler.next_yielded, 5)
        next(iterator)  # advance the iterator again
        self.assertEqual(sampler.yielded, 6)

    def test_drop_last_effect(self):
        num_replicas = 3
        total_samples = len(self.dataset)
        expected_length_with_drop = total_samples // num_replicas
        expected_length_without_drop = math.ceil(total_samples / num_replicas)

        sampler_with_drop = StatefulDistributedSampler(
            self.dataset, num_replicas=3, rank=0, drop_last=True, shuffle=False
        )
        dataloader_with_drop = StatefulDataLoader(self.dataset, sampler=sampler_with_drop)

        sampler_without_drop = StatefulDistributedSampler(
            self.dataset, num_replicas=3, rank=0, drop_last=False, shuffle=False
        )
        dataloader_without_drop = StatefulDataLoader(self.dataset, sampler=sampler_without_drop)

        # Collect all indices from dataloaders
        indices_with_drop = [data for batch in dataloader_with_drop for data in batch]
        indices_without_drop = [data for batch in dataloader_without_drop for data in batch]

        # Check the lengths of the outputs
        self.assertEqual(
            len(indices_with_drop),
            expected_length_with_drop,
            "Length with drop_last=True should match expected truncated length",
        )
        self.assertEqual(
            len(indices_without_drop),
            expected_length_without_drop,
            "Length with drop_last=False should match total dataset size",
        )

        self.assertTrue(
            len(indices_with_drop) <= len(indices_without_drop), "Drop last should result in fewer or equal indices"
        )

    def test_data_order_with_shuffle(self):
        sampler = StatefulDistributedSampler(self.dataset, num_replicas=1, rank=0, shuffle=True)
        indices = list(iter(sampler))
        data_sampled = [self.dataset[i] for i in indices]
        self.assertNotEqual(data_sampled, list(range(100)), "Data should be shuffled")

        dataloader = StatefulDataLoader(self.dataset, sampler=sampler)
        data_loaded = []
        for batch in dataloader:
            data_loaded.extend(batch)
        self.assertEqual(len(data_loaded), len(self.dataset), "All data should be loaded")
        self.assertEqual(data_loaded, data_sampled, "Data loaded by DataLoader should match data sampled by sampler")

    def test_data_order_without_shuffle(self):
        sampler = StatefulDistributedSampler(self.dataset, num_replicas=1, rank=0, shuffle=False)
        indices = list(iter(sampler))
        data_sampled = [self.dataset[i] for i in indices]
        self.assertEqual(data_sampled, list(range(100)), "Data should not be shuffled")

        batch_size = 32
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)
        data_loaded = []
        for batch in dataloader:
            data_loaded.extend(batch)
        self.assertEqual(len(data_loaded), len(self.dataset), "All data should be loaded")
        self.assertEqual(data_loaded, data_sampled, "Data loaded by DataLoader should match data sampled by sampler")
        self.assertEqual(data_loaded, list(range(100)), "Data loaded by DataLoader should be in original order")

    def test_data_distribution_across_replicas(self):
        num_replicas = 5
        all_data = []
        for rank in range(num_replicas):
            sampler = StatefulDistributedSampler(self.dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
            dataloader = torch.utils.data.DataLoader(self.dataset, sampler=sampler)
            data_loaded = []
            for batch in dataloader:
                data_loaded.extend([int(x.item()) for x in batch])
            all_data.extend(data_loaded)
        self.assertEqual(
            sorted(all_data), list(range(100)), "All data points should be covered exactly once across all replicas"
        )


if __name__ == "__main__":
    run_tests()
