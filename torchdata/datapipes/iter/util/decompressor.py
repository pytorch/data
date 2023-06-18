# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bz2
import gzip
import io
import lzma
import os
import pathlib
import sys
import tarfile
import zipfile
import zlib

from enum import Enum
from io import IOBase
from typing import Iterator, Optional, Tuple, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


class CompressionType(Enum):
    GZIP = "gzip"
    LZMA = "lzma"
    TAR = "tar"
    ZIP = "zip"
    BZIP2 = "bz2"
    ZLIB = "zlib"


@functional_datapipe("decompress")
class DecompressorIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes tuples of path and compressed stream of data, and returns tuples of
    path and decompressed stream of data (functional name: ``decompress``). The input compression format can be specified
    or automatically detected based on the files' file extensions.

    Args:
        source_datapipe: IterDataPipe containing tuples of path and compressed stream of data
        file_type: Optional `string` or ``CompressionType`` that represents the compression format of the inputs

    Example:
        >>> from torchdata.datapipes.iter import FileLister, FileOpener
        >>> tar_file_dp = FileLister(self.temp_dir.name, "*.tar")
        >>> tar_load_dp = FileOpener(tar_file_dp, mode="b")
        >>> tar_decompress_dp = Decompressor(tar_load_dp, file_type="tar")
        >>> for _, stream in tar_decompress_dp:
        >>>     print(stream.read())
        b'0123456789abcdef'
    """

    types = CompressionType

    _DECOMPRESSORS = {
        types.GZIP: lambda file: gzip.GzipFile(fileobj=file),
        types.LZMA: lambda file: lzma.LZMAFile(file),
        types.TAR: lambda file: tarfile.open(fileobj=file, mode="r:*"),
        types.ZIP: lambda file: zipfile.ZipFile(file=file),
        types.BZIP2: lambda file: bz2.BZ2File(filename=file),
        types.ZLIB: lambda file: _ZlibFile(file),
    }

    def __init__(
        self, source_datapipe: IterDataPipe[Tuple[str, IOBase]], file_type: Optional[Union[str, CompressionType]] = None
    ) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IOBase]] = source_datapipe
        if isinstance(file_type, str):
            file_type = self.types(file_type.lower())
        self.file_type: Optional[CompressionType] = file_type

    def _detect_compression_type(self, path: str) -> CompressionType:
        if self.file_type:
            return self.file_type

        ext = "".join(pathlib.Path(path).suffixes)
        if ext in {".tar.gz", ".tar.xz"}:
            return self.types.TAR
        else:
            ext = os.path.splitext(path)[1]
        if ext == ".tar":
            return self.types.TAR
        elif ext == ".xz":
            return self.types.LZMA
        elif ext == ".gz":
            return self.types.GZIP
        elif ext == ".zip":
            return self.types.ZIP
        elif ext == ".bz2":
            return self.types.BZIP2
        elif ext == ".zlib":
            return self.types.ZLIB
        else:
            raise RuntimeError(
                f"File at {path} has file extension {ext}, which does not match what are supported by"
                f"ExtractorIterDataPipe."
            )

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for path, file in self.source_datapipe:
            try:
                file_type = self._detect_compression_type(path)
                decompressor = self._DECOMPRESSORS[file_type]
                yield path, StreamWrapper(decompressor(file), file, name=path)
            finally:
                if isinstance(file, StreamWrapper):
                    file.autoclose()


@functional_datapipe("extract")
class ExtractorIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Please use ``Decompressor`` or ``.decompress`` instead.
    """

    def __new__(
        cls, source_datapipe: IterDataPipe[Tuple[str, IOBase]], file_type: Optional[Union[str, CompressionType]] = None
    ):
        return DecompressorIterDataPipe(source_datapipe, file_type)


class _ZlibFile:
    """
    A minimal read-only file object for decompressing zlib data. It's only intended to be wrapped by
    StreamWrapper and isn't intended to be used outside decompressor.py. It only supports the
    specific operations expected by StreamWrapper.
    """

    def __init__(self, file) -> None:
        self._decompressor = zlib.decompressobj()
        self._file = file

        # Stores decompressed bytes leftover from a call to __next__: since __next__ only returns
        # up until the next newline, we need to store the bytes from after the newline for future
        # calls to read or __next__.
        self._buffer = bytearray()

        # Whether or not self._file still has bytes left to read
        self._file_exhausted = False

    def read(self, size: int = -1) -> bytearray:
        if size < 0:
            return self.readall()

        if not size:
            return bytearray()

        result = self._buffer[:size]
        self._buffer = self._buffer[size:]
        while len(result) < size and self._compressed_bytes_remain():
            # If decompress was called previously, there might be some compressed bytes from a previous chunk
            # that haven't been decompressed yet (because decompress() was passed a max_length value that
            # didn't exhaust the compressed bytes in the chunk). We can retrieve these leftover bytes from
            # unconsumed_tail:
            chunk = self._decompressor.unconsumed_tail
            # Let's read compressed bytes in chunks of io.DEFAULT_BUFFER_SIZE because this is what python's gzip
            # library does:
            # https://github.com/python/cpython/blob/a6326972253bf5282c5bf422f4a16d93ace77b57/Lib/gzip.py#L505
            if len(chunk) < io.DEFAULT_BUFFER_SIZE:
                compressed_bytes = self._file.read(io.DEFAULT_BUFFER_SIZE - len(chunk))
                if compressed_bytes:
                    chunk += compressed_bytes
                else:
                    self._file_exhausted = True
            decompressed_chunk = self._decompressor.decompress(chunk, max_length=size - len(result))
            result.extend(decompressed_chunk)
        if not self._compressed_bytes_remain() and not self._decompressor.eof:
            # There are no more compressed bytes available to decompress, but we haven't reached the
            # zlib EOF, so something is wrong.
            raise EOFError("Compressed file ended before the end-of-stream marker was reached")
        return result

    def readall(self):
        """
        This is just mimicking python's internal DecompressReader.readall:
        https://github.com/python/cpython/blob/a6326972253bf5282c5bf422f4a16d93ace77b57/Lib/_compression.py#L113
        """
        chunks = []
        # sys.maxsize means the max length of output buffer is unlimited,
        # so that the whole input buffer can be decompressed within one
        # .decompress() call.
        data = self.read(sys.maxsize)
        while data:
            chunks.append(data)
            data = self.read(sys.maxsize)

        return b"".join(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._buffer and not self._compressed_bytes_remain():
            raise StopIteration

        # Check if the buffer already has a newline, in which case we don't need to do any decompression of
        # remaining bytes
        newline_index = self._buffer.find(b"\n")
        if newline_index != -1:
            line = self._buffer[: newline_index + 1]
            self._buffer = self._buffer[newline_index + 1 :]
            return line

        # Keep decompressing bytes until we find a newline or run out of bytes
        line = self._buffer
        self._buffer = bytearray()
        while self._compressed_bytes_remain():
            decompressed_chunk = self.read(io.DEFAULT_BUFFER_SIZE)
            newline_index = decompressed_chunk.find(b"\n")
            if newline_index == -1:
                line.extend(decompressed_chunk)
            else:
                line.extend(decompressed_chunk[: newline_index + 1])
                self._buffer.extend(decompressed_chunk[newline_index + 1 :])
                return line
        return line

    def _compressed_bytes_remain(self) -> bool:
        """
        True if there are compressed bytes still left to decompress. False otherwise.
        """
        return not self._file_exhausted or self._decompressor.unconsumed_tail != b""
