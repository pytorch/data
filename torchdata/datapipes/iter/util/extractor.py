# Copyright (c) Facebook, Inc. and its affiliates.
import gzip
import lzma
import os
import pathlib
import tarfile
import zipfile
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


@functional_datapipe("extract")
class ExtractorIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Iterable DataPipe that takes tuples of path and compressed stream of data, and return tuples of
    path and decompressed (extracted) stream of data. The input compression format can be specified
    or automatically detected based on the files' file extensions.

    Args:
        source_datapipe: IterDataPipe containing tuples of path and compressed stream of data
        file_type: Optional string or CompressionType that represents what compression format of the inputs
    """

    types = CompressionType

    _DECOMPRESSORS = {
        types.GZIP: lambda file: gzip.GzipFile(fileobj=file),
        types.LZMA: lambda file: lzma.LZMAFile(file),
        types.TAR: lambda file: tarfile.open(fileobj=file, mode="r:*"),
        types.ZIP: lambda file: zipfile.ZipFile(file=file),
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
        else:
            raise RuntimeError(
                f"File at {path} has file extension {ext}, which does not match what are supported by"
                f"ExtractorIterDataPipe."
            )

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for path, file in self.source_datapipe:
            file_type = self._detect_compression_type(path)
            decompressor = self._DECOMPRESSORS[file_type]
            yield path, StreamWrapper(decompressor(file))
