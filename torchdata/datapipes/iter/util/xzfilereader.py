# Copyright (c) Facebook, Inc. and its affiliates.
import lzma
import warnings
from io import BufferedIOBase
from typing import Iterable, Iterator, Tuple

from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("read_from_xz")
class XzFileReaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r"""

    Iterable datapipe to uncompress xz (lzma) binary streams from an input iterable which contains tuples of
    pathname and xy binary streams. This yields a tuple of pathname and extracted binary stream.
    args:
        datapipe: Iterable datapipe that provides tuples of pathname and xy binary stream
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed automatically if the default DecoderDataPipe
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically.
    """

    def __init__(self, datapipe: Iterable[Tuple[str, BufferedIOBase]], length: int = -1):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                extracted_fobj = lzma.open(data_stream, mode="rb")  # type: ignore[call-overload]
                new_pathname = pathname.rstrip(".xz")
                yield new_pathname, extracted_fobj  # type: ignore[misc]
            except Exception as e:
                warnings.warn(f"Unable to extract files from corrupted xz/lzma stream {pathname} due to: {e}, abort!")
                raise e

    def __len__(self):
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length
