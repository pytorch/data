# Copyright (c) Facebook, Inc. and its affiliates.
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
    """

    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO]], **kwargs) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IO]] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Tuple[str, Dict]]:
        for file_name, stream in self.source_datapipe:
            data = stream.read()
            stream.close()
            yield file_name, json.loads(data)

    def __len__(self) -> int:
        return len(self.source_datapipe)
