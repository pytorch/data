# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("header")
class HeaderIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, limit: int = 10):
        self.source_datapipe = source_datapipe
        self.limit = limit

    def __iter__(self):
        for i, value in enumerate(self.source_datapipe):
            if i < self.limit:
                yield value
            else:
                break

    # TODO: Fix the case that the length of source_datapipe is shorter than limit
    def __len__(self):
        return self.limit
