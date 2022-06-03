# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Iterator, Optional, Tuple

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

try:
    import datasets
except ImportError:
    datasets = None


def _get_response_from_huggingface_hub(
    dataset: str, split: str, revision: str, streaming: bool, data_files: Optional[Dict[str, str]]
) -> Iterator[Any]:
    hf_dataset = datasets.load_dataset(
        dataset, split=split, revision=revision, streaming=streaming, data_files=data_files
    )
    return iter(hf_dataset)


class HuggingFaceHubReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes in dataset names and returns an Iterable HuggingFace dataset
    Args format is the same as https://huggingface.co/docs/datasets/loading
    Args:
        source_datapipe: a DataPipe that contains dataset names which will be accepted by the HuggingFace datasets library
        revision: the specific dataset version
        split: train/test split
        streaming: Stream dataset instead of downloading it one go
        data_files: Optional dict to set custom train/test/validation split
    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, HuggingFaceHubReaderIterDataPipe
        >>> huggingface_reader_dp = HuggingFaceHubReaderDataPipe("lhoestq/demo1", revision="main")
        >>> elem = next(iter(huggingface_reader_dp))
        >>> elem["package_name"]
        com.mantz_it.rfanalyzer
    """

    source_datapipe: IterDataPipe[str]

    def __init__(
        self,
        dataset: str,
        *,
        split: str = "train",
        revision: str = "main",
        streaming: bool = True,
        data_files: Optional[Dict[str, str]] = None,
    ) -> None:
        if datasets is None:
            raise ModuleNotFoundError(
                "Package `datasets` is required to be installed to use this datapipe."
                "Please use `pip install datasets` or `conda install -c conda-forge datasets`"
                "to install the package"
            )

        self.dataset = dataset
        self.split = split
        self.revision = revision
        self.streaming = streaming
        self.data_files = data_files

    def __iter__(self) -> Iterator[Any]:
        return _get_response_from_huggingface_hub(
            dataset=self.dataset,
            split=self.split,
            revision=self.revision,
            streaming=self.streaming,
            data_files=self.data_files,
        )

    def __len__(self) -> int:
        return len([self.dataset])
