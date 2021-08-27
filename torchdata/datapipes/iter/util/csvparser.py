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


@functional_datapipe('parse_csv')
class CSVParserIterDataPipe(_CSVBaseParserIterDataPipe):
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
        super().__init__(source_datapipe,
                         csv.reader,
                         skip_header=skip_header,
                         decode=decode,
                         encoding=encoding,
                         errors=errors,
                         keep_filename=keep_filename,
                         **fmtparams)


@functional_datapipe('parse_csv_as_dict')
class CSVDictParserIterDataPipe(_CSVBaseParserIterDataPipe):
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
        super().__init__(source_datapipe,
                         csv.DictReader,
                         skip_header=skip_header,
                         decode=decode,
                         encoding=encoding,
                         errors=errors,
                         keep_filename=keep_filename,
                         **fmtparams)
