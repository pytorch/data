# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data import IterDataPipe, functional_datapipe
from typing import Optional


@functional_datapipe('cycle')
class CyclerIterDataPipe(IterDataPipe):
    """
    Cycle the specified input forever (default), or specified number of times.
    """
    def __init__(self, source_datapipe: IterDataPipe, count: Optional[int] = None):
        self.source_datapipe = source_datapipe
        self.count = count
        if count is not None and count < 0:
            raise ValueError(f"Expected non-negative count, got {count}")

    def __iter__(self):
        i = 0
        while self.count is None or i < self.count:
            for x in self.source_datapipe:
                yield x
            i += 1

    def __len__(self):
        if self.count is None:
            raise TypeError(f"This {type(self).__name__} instance cycles forever,"
                            "and therefore doesn't have valid length")
        else:
            return self.count * len(self.source_datapipe)
