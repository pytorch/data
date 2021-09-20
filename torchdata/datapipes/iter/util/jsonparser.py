# Copyright (c) Facebook, Inc. and its affiliates.
import json

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("parse_json_files")
class JsonParserIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, **kwargs):
        self.source_datapipe = source_datapipe
        self.kwargs = kwargs

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            data = stream.read()
            yield file_name, json.loads(data)

    def __len__(self):
        return len(self.source_datapipe)
