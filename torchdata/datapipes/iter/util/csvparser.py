# Copyright (c) Facebook, Inc. and its affiliates.
import csv

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe('parse_csv_files')
class CSVParserIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, *, skip_header=0, **fmtparams):
        self.source_datapipe = source_datapipe
        self.fmtparams = fmtparams
        self.skip_head = skip_header

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            lines = [bytes_line.decode(errors="ignore")
                     for bytes_line in stream.readlines()]
            reader = csv.reader(lines, **self.fmtparams)
            skip = self.skip_head
            for row in reader:
                if skip > 0:
                    skip -= 1
                    continue
                yield tuple([file_name] + row)
