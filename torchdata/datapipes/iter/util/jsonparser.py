# Copyright (c) Facebook, Inc. and its affiliates.
import json

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from typing import Dict, Tuple, IO


@functional_datapipe("parse_json_files")
class JsonParserIterDataPipe(IterDataPipe[Tuple[str, Dict]]):
    r"""
    Iterable DataPipe that reads from JSON data stream and yields a tuple of file name and JSON data

    Args:
        source_datapipe: a DataPipe with tuples of file name and JSON data stream
        kwargs: keyword arguments that will be passed through to `json.loads`
    """
    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO]], **kwargs) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IO]] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            data = stream.read()
            yield file_name, json.loads(data)

    def __len__(self):
        return len(self.source_datapipe)
