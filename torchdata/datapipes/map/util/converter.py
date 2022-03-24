# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Optional

from torch.utils.data import IterDataPipe, MapDataPipe


# @functional_datapipe("to_iter_datapipe")  # This line must be kept for .pyi signature parser
class MapToIterConverterIterDataPipe(IterDataPipe):
    def __init__(self, datapipe: MapDataPipe, indices: Optional[List] = None):
        if not isinstance(datapipe, MapDataPipe):
            raise TypeError(f"MapToIterConverter can only apply on MapDataPipe, but found {type(datapipe)}")
        self.datapipe: MapDataPipe = datapipe
        self.indices = indices if indices else range(len(datapipe))

    def __iter__(self):
        for idx in self.indices:
            yield self.datapipe[idx]

    def __len__(self):
        return len(self.datapipe)


MapDataPipe.register_datapipe_as_function("to_iter_datapipe", MapToIterConverterIterDataPipe)
