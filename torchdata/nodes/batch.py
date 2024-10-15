from typing import Iterator, List

from torchdata.nodes import BaseNode, T


class Batcher(BaseNode[List[T]]):
    def __init__(self, source: BaseNode[T], batch_size: int, drop_last: bool = True):
        self.source = source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def iterator(self) -> Iterator[List[T]]:
        batch = []
        for item in self.source:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) and not self.drop_last:
            yield batch
