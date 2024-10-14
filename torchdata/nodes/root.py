# pyre-unsafe
from typing import Iterator

from torch._utils import ExceptionWrapper
from torchdata.nodes import BaseNode, T


class Root(BaseNode[T]):
    def __init__(self, source: BaseNode):
        self.source = source

    def iterator(self) -> Iterator[T]:
        for x in self.source:
            if isinstance(x, ExceptionWrapper):
                x.reraise()
            elif isinstance(x, StopIteration):
                raise x
            yield x
