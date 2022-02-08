# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, Iterator, Tuple, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

K = TypeVar("K")


@functional_datapipe("enumerate")
class EnumeratorIterDataPipe(IterDataPipe[Tuple[int, K]]):
    r"""
    Adds an index to an existing DataPipe through enumeration, with
    the index starting from 0 by default (functional name: ``enumerate``).

    Args:
        source_datapipe: Iterable DataPipe being indexed
        starting_index: Index from which enumeration will start
    """

    def __init__(self, source_datapipe: IterDataPipe[K], starting_index: int = 0) -> None:
        self.source_datapipe: IterDataPipe[K] = source_datapipe
        self.starting_index = starting_index

    def __iter__(self):
        yield from enumerate(self.source_datapipe, self.starting_index)

    def __len__(self):
        return len(self.source_datapipe)


@functional_datapipe("add_index")
class IndexAdderIterDataPipe(IterDataPipe[Dict]):
    r"""
    Adds an index to an existing Iterable DataPipe with (functional name: ``add_index``). The row or batch
    within the DataPipe must have the type `Dict`; otherwise, a `NotImplementedError` will be thrown. The index
    of the data is set to the provided ``index_name``.

    Args:
        source_datapipe: Iterable DataPipe being indexed
        index_name: Name of the key to store data index
    """

    def __init__(self, source_datapipe: IterDataPipe[Dict], index_name: str = "index") -> None:
        self.source_datapipe = source_datapipe
        self.index_name = index_name

    def __iter__(self) -> Iterator[Dict]:
        for i, row_or_batch in enumerate(self.source_datapipe):
            if isinstance(row_or_batch, dict):
                row_or_batch[self.index_name] = i
                yield row_or_batch
            else:
                raise NotImplementedError("We only support adding index to row or batch in dict type")

    def __len__(self) -> int:
        return len(self.source_datapipe)
