# Copyright (c) Facebook, Inc. and its affiliates.
import csv
from typing import Tuple, Union, Iterator, TypeVar

from torch.utils.data import IterDataPipe, functional_datapipe

D = TypeVar("D")


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
    ) -> None:
        if skip_lines < 0:
            raise ValueError("'skip_lines' is required to be a positive integer.")
        self._skip_lines = skip_lines
        self._strip_newline = strip_newline
        self._decode = decode
        self._encoding = encoding
        self._errors = errors
        self._return_path = return_path

    def skip_lines(self, stream: Iterator[D]) -> Iterator[D]:
        if self._skip_lines:
            try:
                for _ in range(self._skip_lines):
                    next(stream)
            except StopIteration:
                stream = iter(())

        yield from stream

    def strip_newline(self, stream: Iterator[Union[bytes, str]]) -> Iterator[Union[bytes, str]]:
        if not self._strip_newline:
            yield from stream
            return

        for line in stream:
            if self._strip_newline:
                if isinstance(line, str):
                    yield line.strip("\n")
                else:
                    yield line.strip(b"\n")
            else:
                yield line

    def decode(self, stream: Iterator[Union[str, bytes]]) -> Iterator[Union[str, bytes]]:
        if not self._decode:
            yield from stream
            return

        for line in stream:
            if self._decode and isinstance(line, bytes):
                yield line.decode(self._encoding, self._errors)
            else:
                yield line

    def return_path(self, stream: Iterator[D], *, path: str) -> Iterator[Union[D, Tuple[str, D]]]:
        if not self._return_path:
            yield from stream
            return

        for data in stream:
            if self._return_path:
                yield path, data
            else:
                yield data


@functional_datapipe("readlines")
class LineReaderIterDataPipe(IterDataPipe[Union[Union[str, bytes], Tuple[str, Union[str, bytes]]]]):
    r"""
    Iterable DataPipe that accepts a DataPipe consisting of tuples of file name and string data stream,
    and for each line in the stream, it yields a tuple of file name and the line

    Args:
        source_datapipe: a DataPipe with tuples of file name and string data stream
        skip_lines: number of lines to skip at the beginning of each file
        strip_newline: if True, the new line character will be stripped
        decode: if True, this will decode the contents of the file based on the specified encoding
        encoding: the character encoding of the files (default='utf-8')
        errors: the error handling scheme used while decoding
        return_path: if True, each line will return a tuple of path and contents, rather
            than just the contents
    """

    def __init__(
        self,
        source_datapipe,
        *,
        skip_lines: int = 0,
        strip_newline: bool = True,
        decode: bool = False,
        encoding="utf-8",
        errors: str = "ignore",
        return_path: bool = True,
    ):
        self.source_datapipe = source_datapipe
        self._helper = PlainTextReaderHelper(
            skip_lines=skip_lines,
            strip_newline=strip_newline,
            decode=decode,
            encoding=encoding,
            errors=errors,
            return_path=return_path,
        )

    def __iter__(self):
        for path, stream in self.source_datapipe:
            stream = self._helper.skip_lines(stream)
            stream = self._helper.strip_newline(stream)
            stream = self._helper.decode(stream)
            stream = self._helper.return_path(stream, path=path)
            yield from stream


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
        **fmtparams,
    ):
        self.source_datapipe = source_datapipe
        self._csv_reader = csv_reader
        self._helper = PlainTextReaderHelper(
            skip_lines=skip_lines,
            decode=decode,
            encoding=encoding,
            errors=errors,
            return_path=return_path,
        )
        self.fmtparams = fmtparams

    def __iter__(self):
        for path, stream in self.source_datapipe:
            stream = self._helper.skip_lines(stream)
            stream = self._helper.decode(stream)
            stream = self._csv_reader(stream, **self.fmtparams)
            stream = self._helper.return_path(stream, path=path)
            yield from stream


@functional_datapipe("parse_csv")
class CSVParserIterDataPipe(_CSVBaseParserIterDataPipe):
    r"""
    Iterable DataPipe that accepts a DataPipe consists of tuples of file name and CSV data stream.
    This reads and returns the contents within the CSV files one row at a time (as a List
    by default, depending on fmtparams).

    Args:
        source_datapipe: source DataPipe with tuples of file name and CSV data stream
        skip_lines: number of lines to skip at the beginning of each file
        strip_newline: if True, the new line character will be stripped
        decode: if True, this will decode the contents of the file based on the specified encoding
        encoding: the character encoding of the files (default='utf-8')
        errors: the error handling scheme used while decoding
        return_path: if True, each line will return a tuple of path and contents, rather
            than just the contents
    """

    def __init__(
        self,
        source_datapipe,
        *,
        skip_header=0,
        decode=True,
        encoding="utf-8",
        errors="ignore",
        keep_filename=False,
        **fmtparams,
    ):
        super().__init__(
            source_datapipe,
            csv.reader,
            skip_lines=skip_header,
            decode=decode,
            encoding=encoding,
            errors=errors,
            return_path=keep_filename,
            **fmtparams,
        )


@functional_datapipe("parse_csv_as_dict")
class CSVDictParserIterDataPipe(_CSVBaseParserIterDataPipe):
    r"""
    Iterable DataPipe that accepts a DataPipe consists of tuples of file name and CSV data stream.
    This reads and returns the contents within the CSV files one row at a time (as a Dict by default,
    depedning on fmtparams).
    The first row of each file, unless skipped, will be used as the header; the contents of the header row
    will be used as keys for the Dicts generated from the remaining rows.

    Args:
        source_datapipe: source DataPipe with tuples of file name and CSV data stream
        skip_lines: number of lines to skip at the beginning of each file
        strip_newline: if True, the new line character will be stripped
        decode: if True, this will decode the contents of the file based on the specified encoding
        encoding: the character encoding of the files (default='utf-8')
        errors: the error handling scheme used while decoding
        return_path: if True, each line will return a tuple of path and contents, rather
            than just the contents
    """

    def __init__(
        self,
        source_datapipe,
        *,
        skip_header=0,
        decode=True,
        encoding="utf-8",
        errors="ignore",
        keep_filename=False,
        **fmtparams,
    ):
        super().__init__(
            source_datapipe,
            csv.DictReader,
            skip_lines=skip_header,
            decode=decode,
            encoding=encoding,
            errors=errors,
            return_path=keep_filename,
            **fmtparams,
        )
