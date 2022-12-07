# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os.path
from typing import Iterator, Tuple
from unittest.mock import patch

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple


class RarfilePatcher:
    def __init__(self):
        from rarfile import DirectReader

        unpatched_read = DirectReader._read

        def patched_read(self, cnt=-1):
            self._fd.seek(self._inf.header_offset, 0)
            self._cur = self._parser._parse_header(self._fd)
            self._cur_avail = self._cur.add_size
            return unpatched_read(self, cnt)

        self._patch = patch("rarfile.DirectReader._read", new=patched_read)

    def start(self):
        self._patch.start()

    def stop(self):
        self._patch.stop()


_PATCHED = False


@functional_datapipe("load_from_rar")
class RarArchiveLoaderIterDataPipe(IterDataPipe[Tuple[str, io.BufferedIOBase]]):
    r"""
    Decompresses rar binary streams from input Iterable Datapipes which contains tuples of path name and rar
    binary stream, and yields  a tuple of path name and extracted binary stream (functional name: ``load_from_rar``).

    Note:
        The nested RAR archive is not supported by this DataPipe
        due to the limitation of the archive type. Please extract
        outer RAR archive before reading the inner archive.

    Args:
        datapipe: Iterable DataPipe that provides tuples of path name and rar binary stream
        length: Nominal length of the DataPipe

    Example:
        >>> from torchdata.datapipes.iter import FileLister, FileOpener
        >>> datapipe1 = FileLister(".", "*.rar")
        >>> datapipe2 = FileOpener(datapipe1, mode="b")
        >>> rar_loader_dp = datapipe2.load_from_rar()
        >>> for _, stream in rar_loader_dp:
        >>>     print(stream.read())
        b'0123456789abcdef'
    """

    def __init__(self, datapipe: IterDataPipe[Tuple[str, io.BufferedIOBase]], *, length: int = -1):
        try:
            import rarfile
        except ImportError as error:
            raise ModuleNotFoundError(
                "Package `rarfile` is required to be installed to use this datapipe. "
                "Please use `pip install rarfile` or `conda -c conda-forge install rarfile` to install it."
            ) from error

        # check if at least one system library for reading rar archives is available to be used by rarfile
        rarfile.tool_setup()

        self.datapipe = datapipe
        self.length = length

    def __iter__(self) -> Iterator[Tuple[str, io.BufferedIOBase]]:
        import rarfile

        global _PATCHED
        if not _PATCHED:
            patcher = RarfilePatcher()
            patcher.start()
            _PATCHED = True

        for data in self.datapipe:
            try:
                validate_pathname_binary_tuple(data)
                path, stream = data
                if isinstance(stream, rarfile.RarExtFile) or (
                    isinstance(stream, StreamWrapper) and isinstance(stream.file_obj, rarfile.RarExtFile)
                ):
                    raise ValueError(
                        f"Nested RAR archive is not supported by {type(self).__name__}. Please extract outer archive first."
                    )

                rar = rarfile.RarFile(stream)
                for info in rar.infolist():
                    if info.is_dir():
                        continue

                    inner_path = os.path.join(path, info.filename)
                    file_obj = rar.open(info)
                    yield inner_path, StreamWrapper(file_obj, stream, name=path)  # type: ignore[misc]
            finally:
                if isinstance(stream, StreamWrapper):
                    stream.autoclose()

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
