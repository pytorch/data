# Copyright (c) Facebook, Inc. and its affiliates.
import os
import tarfile
import warnings
from io import BufferedIOBase
from typing import cast, IO, Iterable, Iterator, Optional, Tuple

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple


@functional_datapipe("read_from_tar")
class TarArchiveReaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r"""
    Opens/decompresses tar binary streams from an Iterable DataPipe which contains tuples of path name and
    tar binary stream, and yields a tuple of path name and extracted binary stream (functional name: ``read_from_tar``).

    Args:
        datapipe: Iterable DataPipe that provides tuples of path name and tar binary stream
        mode: File mode used by `tarfile.open` to read file object.
            Mode has to be a string of the form `'filemode[:compression]'`
        length: a nominal length of the DataPipe

    Note:
        The opened file handles will be closed automatically if the default ``DecoderDataPipe``
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically.
    """

    def __init__(self, datapipe: Iterable[Tuple[str, BufferedIOBase]], mode: str = "r:*", length: int = -1) -> None:
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.mode: str = mode
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                # typing.cast is used here to silence mypy's type checker
                tar = tarfile.open(fileobj=cast(Optional[IO[bytes]], data_stream), mode=self.mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(f"failed to extract file {tarinfo.name} from source tarfile {pathname}")
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                    yield inner_pathname, StreamWrapper(extracted_fobj)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!")
                raise e

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
