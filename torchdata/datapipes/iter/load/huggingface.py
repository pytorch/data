# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Iterator, Tuple

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

try:
    import datasets
except ImportError:
    datasets = None


def _get_response_from_huggingface_hub(
    dataset: str,
    streaming: bool = True,
    **config_kwargs,
) -> Iterator[Any]:
    hf_dataset = datasets.load_dataset(path=dataset, streaming=streaming, **config_kwargs)
    return iter(hf_dataset)


class HuggingFaceHubReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes in dataset names and returns an Iterable HuggingFace dataset.
    Please refer to https://huggingface.co/docs/datasets/loading for the meaning and type of each argument.
    Contrary to their implementation, default behavior differs in the following:

    * ``streaming`` is set to ``True``

    Args:
        dataset: path or name of the dataset
        **config_kwargs: additional arguments for ``datasets.load_dataset()``

    Example:

    .. testsetup::

        import datasets
        from torchdata.datapipes.iter import IterableWrapper, HuggingFaceHubReader
        from unittest.mock import MagicMock

        datasets.load_dataset = MagicMock(return_value=datasets.Dataset.from_dict(
            {"id": ["7bd227d9-afc9-11e6-aba1-c4b301cdf627", "7bd22905-afc9-11e6-a5dc-c4b301cdf627" ], "package_name": ["com.mantz_it.rfanalyzer"] * 2}
        ))

    .. testcode::

        huggingface_reader_dp = HuggingFaceHubReader("lhoestq/demo1", revision="main")
        elem = next(iter(huggingface_reader_dp))
        assert elem["package_name"] == "com.mantz_it.rfanalyzer"

    """

    def __init__(
        self,
        dataset: str,
        **config_kwargs,
    ) -> None:
        if datasets is None:
            raise ModuleNotFoundError(
                "Package `datasets` is required to be installed to use this datapipe."
                "Please use `pip install datasets` or `conda install -c conda-forge datasets`"
                "to install the package"
            )

        self.dataset = dataset
        self.config_kwargs = config_kwargs

    def __iter__(self) -> Iterator[Any]:
        return _get_response_from_huggingface_hub(dataset=self.dataset, **self.config_kwargs)

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
