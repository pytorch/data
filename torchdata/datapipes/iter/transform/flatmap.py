# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data import DataChunk
from torch.utils.data import functional_datapipe, IterDataPipe

from typing import TypeVar

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("flatmap")
class FlatMapDataPipe(IterDataPipe[IterDataPipe[DataChunk[T_co]]]):

    r""":class:`FlatMapDataPipe`.

    Iterable DataPipe which applies a function to an IterableDataPipe containing
    IterableDataPipes and flattens to a single unnested IterableDataPipe.

    Args:
        datapipe: Iterable datapipe containing iterable datapipes to which the function is applied
        fn: the function to be applied to each of the ``inner" datapipes
    """
    def __init__(self, datapipe, fn):
        self.datapipe = datapipe
        self.fn = fn

    def __iter__(self):
        for e in self.datapipe:
            yield from e.map(self.fn)
