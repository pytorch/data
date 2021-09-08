# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("readlines")
class LineReaderIterDataPipe(IterDataPipe[Tuple[str, str]]):
    r"""
    Iterable DataPipe that accepts a DataPipe consisting of tuples of file name and string data stream,
    and for each line in the stream, it yields a tuple of file name and the line

    Args:
        source_datapipe: a DataPipe with tuples of file name and string data stream
        strip_newline: if True, the new line character ('\n') will be stripped
    """
    def __init__(self, source_datapipe, strip_newline=True):
        self.source_datapipe = source_datapipe
        self.strip_newline = strip_newline

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            for line in stream:
                if self.strip_newline:
                    line = line.rstrip("\n")
                yield file_name, line
