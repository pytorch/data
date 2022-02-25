# Copyright (c) Facebook, Inc. and its affiliates.
import os

from typing import Any, Callable, Iterator, Optional, Tuple, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

U = Union[bytes, bytearray, str]


@functional_datapipe("save_to_disk")
class SaverIterDataPipe(IterDataPipe[str]):
    r"""
    Takes in a DataPipe of tuples of metadata and data, saves the data
    to the target path generated by the ``filepath_fn`` and metadata, and yields file path on local file
    system (functional name: ``save_to_disk``).

    Args:
        source_datapipe: Iterable DataPipe with tuples of metadata and data
        mode: Node in which the file will be opened for write the data (``"w"`` by default)
        filepath_fn: Function that takes in metadata and returns the target path of the new file

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> import os
        >>> def filepath_fn(name: str) -> str:
        >>>     return os.path.join(".", os.path.basename(name))
        >>> name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2", "3.txt": b"DATA3"}
        >>> source_dp = IterableWrapper(sorted(name_to_data.items()))
        >>> saver_dp = source_dp.save_to_disk(filepath_fn=filepath_fn, mode="wb")
        >>> res_file_paths = list(saver_dp)
        >>> res_file_paths
        ['./1.txt', './2.txt', './3.txt']
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[Any, U]],
        mode: str = "w",
        filepath_fn: Optional[Callable] = None,
    ):
        self.source_datapipe: IterDataPipe[Tuple[Any, U]] = source_datapipe
        self.mode: str = mode if "w" in mode else "w" + mode
        self.fn: Optional[Callable] = filepath_fn

    def __iter__(self) -> Iterator[str]:
        for filepath, data in self.source_datapipe:
            if self.fn is not None:
                filepath = self.fn(filepath)
            dirname = os.path.dirname(filepath)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(filepath, self.mode) as f:
                f.write(data)
            yield filepath

    def __len__(self) -> int:
        return len(self.source_datapipe)
