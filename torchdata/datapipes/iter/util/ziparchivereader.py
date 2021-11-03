# Copyright (c) Meta Platforms, Inc. and its affiliates.
import os
import sys
import warnings
import zipfile
from io import BufferedIOBase
from typing import IO, Iterable, Iterator, Tuple, cast

from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("read_from_zip")
class ZipArchiveReaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r""":class:`ZipArchiveReaderIterDataPipe`.

    Iterable DataPipe to extract zip binary streams from input iterable which contains a tuple of path name and
    zip binary stream. This yields a tuple of path name and extracted binary stream.

    Args:
        datapipe: Iterable DataPipe that provides tuples of path name and zip binary stream
        length: Nominal length of the DataPipe

    Note:
        The opened file handles will be closed automatically if the default DecoderDataPipe
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically. Due to how zipfiles implements its open() method,
        the data_stream variable below cannot be closed within the scope of this function.
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
                # typing.cast is used here to silence mypy's type checker
                zips = zipfile.ZipFile(cast(IO[bytes], data_stream))
                for zipinfo in zips.infolist():
                    # major version should always be 3 here.
                    if sys.version_info[1] >= 6:
                        if zipinfo.is_dir():
                            continue
                    elif zipinfo.filename.endswith("/"):
                        continue
                    extracted_fobj = zips.open(zipinfo)
                    inner_pathname = os.path.normpath(os.path.join(pathname, zipinfo.filename))
                    yield inner_pathname, StreamWrapper(extracted_fobj)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(f"Unable to extract files from corrupted zipfile stream {pathname} due to: {e}, abort!")
                raise e
            # We are unable to close 'data_stream' here, because it needs to be available to use later

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
