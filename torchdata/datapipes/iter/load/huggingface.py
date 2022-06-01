# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import posixpath

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from torch.utils.data.datapipes.utils.common import match_masks

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

try:
    import datasets
except ImportError:
    datasets = None

def _get_response_from_huggingface_hub(dataset, split, revision, data_files) -> Tuple[Any, StreamWrapper]:
    dataset = datasets.load_dataset(dataset, split, revision, data_files)
    return dataset[0], StreamWrapper(dataset)

@functional_datapipe("read_from_huggingface_hub")
class HuggingFaceHubReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes in dataset names and returns an Iterable HuggingFace dataset
    Args:
        source_datapipe: a DataPipe that contains dataset names which will be accepted by the HuggingFace datasets library
        revision: the specific dataset version
        data_files: Optional dict to set custom train/test/validation split
    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, HuggingFaceHubReaderIterDataPipe
        >>> huggingface_reader_dp = HuggingFaceHubReaderDataPipe(IterableWrapper(["lhoestq/demo1"]), revision="main")
        >>> reader_dp = huggingface_reader_dp
        >>> it = iter(reader_dp)
        >>> path, line = next(it)
        >>> path
        Add test result here
        >>> line
        Add test result here b'BSD 3-Clause License'
    """

    source_datapipe: IterDataPipe[str]

    def __init__(self, dataset_name: str, *, split : str = "train", revision : Optional[str] = None, data_files : Optional[Dict[str,str]] = None) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.revision = revision
        self.data_files = data_files

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        yield _get_response_from_huggingface_hub(dataset=self.dataset_name, split=self.split, revision=self.revision, data_files=self.data_files)

    def __len__(self) -> int:
        return len(self.source_datapipe)
