# pyre-unsafe
from torch._utils import ExceptionWrapper
from torchdata.nodes import BaseNode


class Root[T](BaseNode[T]):
    def __init__(self, source: BaseNode):
        self.source = source

    def iterator(self):
        for x in self.source:
            if isinstance(x, ExceptionWrapper):
                x.reraise()
            elif isinstance(x, StopIteration):
                raise x
            yield x
