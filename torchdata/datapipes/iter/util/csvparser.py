# Copyright (c) Facebook, Inc. and its affiliates.
import csv

from torch.utils.data import IterDataPipe, functional_datapipe


class _CSVBaseParserIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe,
        csv_reader,
        *,
        skip_header=0,
        decode=True,
        encoding="utf-8",
        errors="ignore",
        keep_filename=False,
        **fmtparams
    ):
        self.source_datapipe = source_datapipe
        self.csv_reader = csv_reader
        self.decode = decode
        self.encoding = encoding
        self.errors = errors
        if skip_header < 0:
            raise ValueError("'skip_header' is required to be a positive integer.")
        self.skip_header: int = skip_header
        self.keep_filename: bool = keep_filename
        self.fmtparams = fmtparams

    def _decode(self, stream):
        for line in stream:
            yield line.decode(self.encoding, self.errors)

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            skip = self.skip_header
            while skip > 0:
                stream.readline()
                skip -= 1
            if self.decode:
                stream = self._decode(stream)
            reader = self.csv_reader(stream, **self.fmtparams)
            for row in reader:
                if self.keep_filename:
                    yield file_name, row
                else:
                    yield row


@functional_datapipe("parse_csv")
class CSVParserIterDataPipe(_CSVBaseParserIterDataPipe):
    r"""
    Iterable DataPipe that accepts a DataPipe consists of tuples of file name and CSV data stream.
    This reads and returns the contents within the CSV files one row at a time (as a List
    by default, depending on fmtparams).

    Args:
        source_datapipe: source DataPipe with tuples of file name and CSV data stream
        skip_header: number of rows to skip at the beginning of each file
        decode: if True, this will decode the contents of the file based on the specified encoding
        encoding: the character encoding of the files (default='utf-8')
        errors: the error handling scheme used while decoding
        keep_filename: if True, each row will return a tuple of file name and contents, rather
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
        **fmtparams
    ):
        super().__init__(
            source_datapipe,
            csv.reader,
            skip_header=skip_header,
            decode=decode,
            encoding=encoding,
            errors=errors,
            keep_filename=keep_filename,
            **fmtparams
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
        skip_header: number of rows to skip at the beginning of each file
        decode: if True, this will decode the contents of the file based on the specified encoding
        encoding: the character encoding of the files (default='utf-8')
        errors: the error handling scheme used while decoding
        keep_filename: if True, each row will return a tuple of file name and contents, rather
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
        **fmtparams
    ):
        super().__init__(
            source_datapipe,
            csv.DictReader,
            skip_header=skip_header,
            decode=decode,
            encoding=encoding,
            errors=errors,
            keep_filename=keep_filename,
            **fmtparams
        )
