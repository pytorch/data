import random
import time
from typing import Iterator

import torch
from torchdata.nodes import BaseNode


class MockSource(BaseNode[dict]):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def iterator(self) -> Iterator[dict]:
        for i in range(self.num_samples):
            yield {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"}


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
