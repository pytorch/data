# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
from typing import Iterator

import torch
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.base_node import BaseNode


class MockGenerator:
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            yield {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"}


def MockSource(num_samples: int) -> BaseNode[dict]:
    return IterableWrapper(MockGenerator(num_samples))


def udf_raises(item):
    raise ValueError("test exception")


class RandomSleepUdf:
    def __init__(self, sleep_max_sec: float = 0.01) -> None:
        self.sleep_max_sec = sleep_max_sec

    def __call__(self, x):
        time.sleep(random.random() * self.sleep_max_sec)
        return x


class Collate:
    def __call__(self, x):
        result = {}
        for k in x[0].keys():
            result[k] = [i[k] for i in x]
        return result


class IterInitError(BaseNode[int]):
    def __init__(self, msg: str = "Iter Init Error") -> None:
        self.msg = msg

    def iterator(self) -> Iterator[int]:
        raise ValueError(self.msg)


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[dict]:
        for i in range(self.num_samples):
            yield {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"}


class DummyMapDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> dict:
        return {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"}


def run_test_save_load_state(test, x: BaseNode, midpoint: int):
    it = iter(x)
    results = []
    for _ in range(midpoint):
        results.append(next(it))
    state_dict = x.state_dict()
    for val in it:
        results.append(val)

    x.load_state_dict(state_dict)
    results_after = list(x)

    test.assertEqual(results_after, results[midpoint:])

    full_results = list(x)
    test.assertEqual(full_results, results)
