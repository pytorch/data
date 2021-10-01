# Copyright (c) Facebook, Inc. and its affiliates.
import os
import tarfile
import warnings
from io import BufferedIOBase
from typing import IO, Iterable, Iterator, Optional, Tuple, cast

from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
from torchdata.datapipes.iter.util._archivereader import _ArchiveReaderIterDataPipe
from torch.utils.data import functional_datapipe


@functional_datapipe("read_from_tar")
class TarArchiveReaderIterDataPipe(_ArchiveReaderIterDataPipe):
    r""":class:`TarArchiveReaderIterDataPipe`.

    Iterable DataPipe to extract tar binary streams from input iterable which contains tuples of path name and
    tar binary stream. This yields a tuple of path name and extracted binary stream.

    Args:
        datapipe: Iterable DataPipe that provides tuples of path name and tar binary stream
        mode: File mode used by `tarfile.open` to read file object.
            Mode has to be a string of the form 'filemode[:compression]'
        length: a nominal length of the DataPipe

    Note:
        The opened file handles will be closed automatically if the default DecoderDataPipe
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically.
    """

    def __init__(self, datapipe: Iterable[Tuple[str, BufferedIOBase]], mode: str = "r:*", length: int = -1):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.mode: str = mode
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            self.open_source_streams[pathname].append(data_stream)
            folder_name = os.path.dirname(pathname)
            try:
                # typing.cast is used here to silence mypy's type checker
                tar = tarfile.open(fileobj=cast(Optional[IO[bytes]], data_stream), mode=self.mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn("failed to extract file {} from source tarfile {}".format(tarinfo.name, pathname))
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(folder_name, tarinfo.name))
                    self.open_archive_streams[inner_pathname].append(extracted_fobj)
                    yield (inner_pathname, extracted_fobj)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(
                    "Unable to extract files from corrupted tarfile stream {} due to: {}, abort!".format(pathname, e)
                )
                raise e

    def __len__(self):
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length
