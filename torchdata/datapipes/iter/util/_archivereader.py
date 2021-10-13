# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from io import BufferedIOBase
from typing import DefaultDict, List, Tuple, Union, IO

from torch.utils.data import IterDataPipe


class _ArchiveReaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    def __init__(self):
        super().__init__()
        self._open_source_streams: DefaultDict[str, List[Union[BufferedIOBase, IO[bytes]]]] = defaultdict(list)

    def _close_all_source_streams(self):
        for _name, streams_list in self._open_source_streams.items():
            for stream in streams_list:
                stream.close()
        self._open_source_streams = defaultdict(list)

    def __del__(self):
        self._close_all_source_streams()
