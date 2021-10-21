# Copyright (c) Facebook, Inc. and its affiliates.
from io import IOBase
from typing import Any, Callable, Tuple

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from torchdata.datapipes.utils.common import _default_filepath_fn


@functional_datapipe("save_to_disk")
class SaverIterDataPipe(IterDataPipe[str]):
    r"""
    Iterable DataPipe that takes in a DataPipe of tuples of metadata and data, saves the data
    to the target path (generated by the filepath_fn and metadata), and yields the resulting path

    Args:
        source_datapipe: a DataPipe with tuples of metadata and data
        mode: mode in which the file will be opened for write the data
        filepath_fn: a function that takes in metadata nad returns the target path of the new file
    """
    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[Any, IOBase]],
        mode: str = "wb",
        filepath_fn: Callable = _default_filepath_fn,
    ):
        self.source_datapipe: IterDataPipe[Tuple[Any, IOBase]] = source_datapipe
        self.mode: str = mode
        self.fn: Callable = filepath_fn

    def __iter__(self):
        for meta, data in self.source_datapipe:
            filepath = self.fn(meta)
            with open(filepath, self.mode) as f:
                f.write(data)
            yield filepath

    def __len__(self):
        return len(self.source_datapipe)
