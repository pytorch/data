# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from io import BufferedIOBase
from typing import DefaultDict, List, Tuple, Union, IO

from torch.utils.data import IterDataPipe


class _ArchiveReaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    def __init__(self):
        super().__init__()
        self.open_source_streams: DefaultDict[str, List[Union[BufferedIOBase, IO[bytes]]]] = defaultdict(list)
        self.open_archive_streams: DefaultDict[str, List[Union[BufferedIOBase, IO[bytes]]]] = defaultdict(list)

    def close_all_source_streams(self):
        for _name, streams_list in self.open_source_streams.items():
            for stream in streams_list:
                stream.close()
        self.open_source_streams = defaultdict(list)

    def close_all_archive_streams(self):
        for _name, stream_list in self.open_archive_streams.items():
            for stream in stream_list:
                stream.close()
        self.open_archive_streams = defaultdict(list)

    def close_all_streams(self):
        self.close_all_source_streams()
        self.close_all_archive_streams()
