# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import IO

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, MapTemplate


@functional_datapipe("parse_json_files")
class JsonParserIterDataPipe(MapTemplate):
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

    def __init__(self, source_datapipe: IterDataPipe, input_col=1, output_col=None, **kwargs) -> None:
        super().__init__(source_datapipe, input_col=input_col, output_col=output_col)
        self.kwargs = kwargs

    def _map(self, stream: IO):
        try:
            data = stream.read()
            return json.loads(data, **self.kwargs)
        finally:
            stream.close()
