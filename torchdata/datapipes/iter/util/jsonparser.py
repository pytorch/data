# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Dict, IO, Iterator, Tuple

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("parse_json_files")
class JsonParserIterDataPipe(IterDataPipe[Tuple[str, Dict]]):
    r"""
    Reads from JSON data streams and yields a tuple of file name and JSON data (functional name: ``parse_json_files``).

    Args:
        source_datapipe: a DataPipe with tuples of file name and JSON data stream
        kwargs: keyword arguments that will be passed through to ``json.loads``

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, FileOpener
        >>> import os
        >>> def get_name(path_and_stream):
        >>>     return os.path.basename(path_and_stream[0]), path_and_stream[1]
        >>> datapipe1 = IterableWrapper(["empty.json", "1.json", "2.json"])
        >>> datapipe2 = FileOpener(datapipe1, mode="b")
        >>> datapipe3 = datapipe2.map(get_name)
        >>> json_dp = datapipe3.parse_json_files()
        >>> list(json_dp)
        [('1.json', ['foo', {'bar': ['baz', None, 1.0, 2]}]), ('2.json', {'__complex__': True, 'real': 1, 'imag': 2})]
    """

    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO]], **kwargs) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IO]] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Tuple[str, Dict]]:
        for file_name, stream in self.source_datapipe:
            data = stream.read()
            stream.close()
            yield file_name, json.loads(data, **self.kwargs)

    def __len__(self) -> int:
        return len(self.source_datapipe)
