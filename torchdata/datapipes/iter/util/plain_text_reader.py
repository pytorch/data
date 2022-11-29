# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import csv
from typing import IO, Iterator, Tuple, TypeVar, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

D = TypeVar("D")
Str_Or_Bytes = Union[str, bytes]


class PlainTextReaderHelper:
    def __init__(
        self,
        *,
        skip_lines: int = 0,
        strip_newline: bool = True,
        decode: bool = True,
        encoding="utf-8",
        errors: str = "ignore",
        return_path: bool = False,
        as_tuple: bool = False,
    ) -> None:
        if skip_lines < 0:
            raise ValueError("'skip_lines' is required to be a positive integer.")
        self._skip_lines = skip_lines
        self._strip_newline = strip_newline
        self._decode = decode
        self._encoding = encoding
        self._errors = errors
        self._return_path = return_path
        self._as_tuple = as_tuple

    def skip_lines(self, file: IO) -> Union[Iterator[bytes], Iterator[str]]:
        with contextlib.suppress(StopIteration):
            for _ in range(self._skip_lines):
                next(file)
        try:
            yield from file
        finally:
            file.close()

    def strip_newline(self, stream: Union[Iterator[bytes], Iterator[str]]) -> Union[Iterator[bytes], Iterator[str]]:
        if not self._strip_newline:
            yield from stream
            return

        for line in stream:
            if isinstance(line, str):
                yield line.strip("\r\n")
            else:
                yield line.strip(b"\r\n")

    def decode(self, stream: Union[Iterator[bytes], Iterator[str]]) -> Union[Iterator[bytes], Iterator[str]]:
        if not self._decode:
            yield from stream
        else:
            for line in stream:
                yield line.decode(self._encoding, self._errors) if isinstance(line, bytes) else line

    def return_path(self, stream: Iterator[D], *, path: str) -> Iterator[Union[D, Tuple[str, D]]]:
        if not self._return_path:
            yield from stream
            return
        for data in stream:
            yield path, data

    def as_tuple(self, stream: Iterator[D]) -> Iterator[Union[D, Tuple]]:
        if not self._as_tuple:
            yield from stream
            return
        for data in stream:
            if isinstance(data, list):
                yield tuple(data)
            else:
                yield data


@functional_datapipe("readlines")
class LineReaderIterDataPipe(IterDataPipe[Union[Str_Or_Bytes, Tuple[str, Str_Or_Bytes]]]):
    r"""
    Accepts a DataPipe consisting of tuples of file name and string data stream, and for each line in the
    stream, yields a tuple of file name and the line (functional name: ``readlines``).

    Args:
        source_datapipe: a DataPipe with tuples of file name and string data stream
        skip_lines: number of lines to skip at the beginning of each file
        strip_newline: if ``True``, the new line character will be stripped
        decode: if ``True``, this will decode the contents of the file based on the specified ``encoding``
        encoding: the character encoding of the files (`default='utf-8'`)
        errors: the error handling scheme used while decoding
        return_path: if ``True``, each line will return a tuple of path and contents, rather
            than just the contents

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> import io
        >>> text1 = "Line1\nLine2"
        >>> text2 = "Line2,1\r\nLine2,2\r\nLine2,3"
        >>> source_dp = IterableWrapper([("file1", io.StringIO(text1)), ("file2", io.StringIO(text2))])
        >>> line_reader_dp = source_dp.readlines()
        >>> list(line_reader_dp)
        [('file1', 'Line1'), ('file1', 'Line2'), ('file2', 'Line2,1'), ('file2', 'Line2,2'), ('file2', 'Line2,3')]
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[str, IO]],
        *,
        skip_lines: int = 0,
        strip_newline: bool = True,
        decode: bool = False,
        encoding="utf-8",
        errors: str = "ignore",
        return_path: bool = True,
    ) -> None:
        self.source_datapipe = source_datapipe
        self._helper = PlainTextReaderHelper(
            skip_lines=skip_lines,
            strip_newline=strip_newline,
            decode=decode,
            encoding=encoding,
            errors=errors,
            return_path=return_path,
        )

    def __iter__(self) -> Iterator[Union[Str_Or_Bytes, Tuple[str, Str_Or_Bytes]]]:
        for path, file in self.source_datapipe:
            stream = self._helper.skip_lines(file)
            stream = self._helper.strip_newline(stream)
            stream = self._helper.decode(stream)
            yield from self._helper.return_path(stream, path=path)  # type: ignore[misc]


class _CSVBaseParserIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe,
        csv_reader,
        *,
        skip_lines: int = 0,
        decode: bool = False,
        encoding="utf-8",
        errors: str = "ignore",
        return_path: bool = True,
        as_tuple: bool = False,
        **fmtparams,
    ) -> None:
        self.source_datapipe = source_datapipe
        self._csv_reader = csv_reader
        self._helper = PlainTextReaderHelper(
            skip_lines=skip_lines,
            decode=decode,
            encoding=encoding,
            errors=errors,
            return_path=return_path,
            as_tuple=as_tuple,
        )
        self.fmtparams = fmtparams

    def __iter__(self) -> Iterator[Union[D, Tuple[str, D]]]:
        for path, file in self.source_datapipe:
            stream = self._helper.skip_lines(file)
            stream = self._helper.decode(stream)
            stream = self._csv_reader(stream, **self.fmtparams)
            stream = self._helper.as_tuple(stream)  # type: ignore[assignment]
            yield from self._helper.return_path(stream, path=path)  # type: ignore[misc]


@functional_datapipe("parse_csv")
class CSVParserIterDataPipe(_CSVBaseParserIterDataPipe):
    r"""
    Accepts a DataPipe consists of tuples of file name and CSV data stream,
    reads and returns the contents within the CSV files one row at a time (functional name: ``parse_csv``).
    Each output is a `List` by default, but it depends on ``fmtparams``.

    Args:
        source_datapipe: source DataPipe with tuples of file name and CSV data stream
        skip_lines: number of lines to skip at the beginning of each file
        strip_newline: if ``True``, the new line character will be stripped
        decode: if ``True``, this will decode the contents of the file based on the specified ``encoding``
        encoding: the character encoding of the files (`default='utf-8'`)
        errors: the error handling scheme used while decoding
        return_path: if ``True``, each line will return a tuple of path and contents, rather
            than just the contents
        as_tuple: if ``True``, each line will return a tuple instead of a list

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, FileOpener
        >>> import os
        >>> def get_name(path_and_stream):
        >>>     return os.path.basename(path_and_stream[0]), path_and_stream[1]
        >>> datapipe1 = IterableWrapper(["1.csv", "empty.csv", "empty2.csv"])
        >>> datapipe2 = FileOpener(datapipe1, mode="b")
        >>> datapipe3 = datapipe2.map(get_name)
        >>> csv_parser_dp = datapipe3.parse_csv()
        >>> list(csv_parser_dp)
        [['key', 'item'], ['a', '1'], ['b', '2'], []]
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[str, IO]],
        *,
        skip_lines: int = 0,
        decode: bool = True,
        encoding: str = "utf-8",
        errors: str = "ignore",
        return_path: bool = False,
        as_tuple: bool = False,
        **fmtparams,
    ) -> None:
        super().__init__(
            source_datapipe,
            csv.reader,
            skip_lines=skip_lines,
            decode=decode,
            encoding=encoding,
            errors=errors,
            return_path=return_path,
            as_tuple=as_tuple,
            **fmtparams,
        )


@functional_datapipe("parse_csv_as_dict")
class CSVDictParserIterDataPipe(_CSVBaseParserIterDataPipe):
    r"""
    Accepts a DataPipe consists of tuples of file name and CSV data stream, reads and returns the contents
    within the CSV files one row at a time (functional name: ``parse_csv_as_dict``).

    Each output is a `Dict` by default, but it depends on ``fmtparams``. The first row of each file, unless skipped,
    will be used as the header; the contents of the header row will be used as keys for the `Dict`\s
    generated from the remaining rows.

    Args:
        source_datapipe: source DataPipe with tuples of file name and CSV data stream
        skip_lines: number of lines to skip at the beginning of each file
        strip_newline: if ``True``, the new line character will be stripped
        decode: if ``True``, this will decode the contents of the file based on the specified ``encoding``
        encoding: the character encoding of the files (`default='utf-8'`)
        errors: the error handling scheme used while decoding
        return_path: if ``True``, each line will return a tuple of path and contents, rather
            than just the contents

    Example:
        >>> from torchdata.datapipes.iter import FileLister, FileOpener
        >>> import os
        >>> def get_name(path_and_stream):
        >>>     return os.path.basename(path_and_stream[0]), path_and_stream[1]
        >>> datapipe1 = FileLister(".", "*.csv")
        >>> datapipe2 = FileOpener(datapipe1, mode="b")
        >>> datapipe3 = datapipe2.map(get_name)
        >>> csv_dict_parser_dp = datapipe3.parse_csv_as_dict()
        >>> list(csv_dict_parser_dp)
        [{'key': 'a', 'item': '1'}, {'key': 'b', 'item': '2'}]
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[str, IO]],
        *,
        skip_lines: int = 0,
        decode: bool = True,
        encoding: str = "utf-8",
        errors: str = "ignore",
        return_path: bool = False,
        **fmtparams,
    ) -> None:
        super().__init__(
            source_datapipe,
            csv.DictReader,
            skip_lines=skip_lines,
            decode=decode,
            encoding=encoding,
            errors=errors,
            return_path=return_path,
            **fmtparams,
        )
