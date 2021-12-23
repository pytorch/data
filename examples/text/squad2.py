# Copyright (c) Facebook, Inc. and its affiliates.
import os

from torchdata.datapipes.iter import (
    FileOpener,
    HttpReader,
    IterDataPipe,
    IterableWrapper,
)
from .utils import (
    _add_docstring_header,
    _create_dataset_directory,
    _wrap_split_argument,
)

URL = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}

MD5 = {
    "train": "62108c273c268d70893182d5cf8df740",
    "dev": "246adae8b7002f8679c027697b0b7cf8",
}

NUM_LINES = {
    "train": 130319,
    "dev": 11873,
}


DATASET_NAME = "SQuAD2"


class _ParseSQuADQAData(IterDataPipe):
    def __init__(self, source_datapipe) -> None:
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for _, stream in self.source_datapipe:
            raw_json_data = stream["data"]
            for layer1 in raw_json_data:
                for layer2 in layer1["paragraphs"]:
                    for layer3 in layer2["qas"]:
                        _context, _question = layer2["context"], layer3["question"]
                        _answers = [item["text"] for item in layer3["answers"]]
                        _answer_start = [item["answer_start"] for item in layer3["answers"]]
                        if len(_answers) == 0:
                            _answers = [""]
                            _answer_start = [-1]
                        yield (_context, _question, _answers, _answer_start)


@_add_docstring_header(num_lines=NUM_LINES)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "dev"))
def SQuAD2(root, split):
    """Demonstrates use case when more complex processing is needed on data-stream
    Here we process dictionary returned by standard JSON reader and write custom
        datapipe to orchestrates data samples for Q&A use-case
    """

    url_dp = IterableWrapper([URL[split]])
    # cache data on-disk with sanity check
    cache_dp = url_dp.on_disk_cache(filepath_fn=lambda x: os.path.join(root, os.path.basename(x)), hash_dict={os.path.join(root, os.path.basename(URL[split])): MD5[split]}, hash_type="md5")
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_dp = FileOpener(cache_dp, mode='b')

    # stack custom data pipe on top of JSON reader to orchestrate data samples for Q&A dataset
    return _ParseSQuADQAData(cache_dp.parse_json_files())
