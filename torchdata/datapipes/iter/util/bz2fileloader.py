# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bz2
import warnings
from io import BufferedIOBase
from typing import Iterable, Iterator, Tuple

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple


@functional_datapipe("load_from_bz2")
class Bz2FileLoaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r"""
    Decompresses bz2 binary streams from an Iterable DataPipe which contains tuples of
    path name and bz2 binary streams, and yields a tuple of path name and extracted binary
    stream (functional name: ``load_from_bz2``).

    Args:
        datapipe: Iterable DataPipe that provides tuples of path name and bz2 binary stream
        length: Nominal length of the DataPipe

    Note:
        The opened file handles will be closed automatically if the default ``DecoderDataPipe``
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically.

    Example:
        >>> from torchdata.datapipes.iter import FileLister, FileOpener
        >>> datapipe1 = FileLister(".", "*.bz2")
        >>> datapipe2 = FileOpener(datapipe1, mode="b")
        >>> bz2_loader_dp = datapipe2.load_from_bz2()
        >>> for _, stream in bz2_loader_dp:
        >>>     print(stream.read())
        b'0123456789abcdef'
    """

    def __init__(self, datapipe: Iterable[Tuple[str, BufferedIOBase]], length: int = -1) -> None:
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                extracted_fobj = bz2.open(data_stream, mode="rb")  # type: ignore[call-overload]
                new_pathname = pathname.rstrip(".bz2")
                yield new_pathname, StreamWrapper(extracted_fobj, data_stream, name=new_pathname)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(f"Unable to extract files from corrupted bzip2 stream {pathname} due to: {e}, abort!")
                raise e
            finally:
                if isinstance(data_stream, StreamWrapper):
                    data_stream.autoclose()

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
