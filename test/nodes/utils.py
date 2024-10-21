from typing import Iterator

import torch
from torchdata.nodes import BaseNode


class MockSource(BaseNode[int]):
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def iterator(self) -> Iterator[int]:
        for i in range(self.num_samples):
            yield {"step": i, "test_tensor": torch.tensor([i]), "test_str": f"str_{i}"}


def udf_raises(item):
    raise ValueError("test exception")


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
