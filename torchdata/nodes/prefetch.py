# pyre-unsafe
from typing import Iterator

from torchdata.nodes import BaseNode, T

from torchdata.nodes.map import _SingleThreadedMapper

from ._populate_queue import _populate_queue


class Prefetcher(BaseNode[T]):
    def __init__(self, source: BaseNode[T], prefetch_factor: int):
        self.source = source
        self.prefetch_factor = prefetch_factor

    def iterator(self) -> Iterator[T]:
        return _SingleThreadedMapper(
            source=self.source,
            prefetch_factor=self.prefetch_factor,
            worker=_populate_queue,
        )
