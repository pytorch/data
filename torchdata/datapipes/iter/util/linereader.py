# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe('readlines')
class LineReaderIterDataPipe(IterDataPipe[Tuple[str, str]]):
    """
    Given a pipe containing (filename, stream) read from the stream, and yield filename and each line.
    """
    def __init__(self, source_datapipe, strip_newline=True):
        self.source_datapipe = source_datapipe
        self.strip_newline = strip_newline

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            for line in stream:
                if self.strip_newline:
                    line = line.rstrip(b'\n')
                yield (file_name, line)
