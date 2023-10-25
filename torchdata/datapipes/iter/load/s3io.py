# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import Iterator, List, Tuple, Union

import torchdata

from torch.utils.data.datapipes.utils.common import match_masks
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("list_files_by_s3")
class S3FileListerIterDataPipe(IterDataPipe[str]):
    r"""
    Iterable DataPipe that lists Amazon S3 file URLs with the given prefixes (functional name: ``list_files_by_s3``).
    Acceptable prefixes include ``s3://bucket-name``, ``s3://bucket-name/``, ``s3://bucket-name/folder``.

    Note:
        1. ``source_datapipe`` **must** contain a list of valid S3 URLs
        2. ``length`` is `-1` by default, and any call to ``__len__()`` is invalid, because the length is unknown
           until all files are iterated.
        3. ``request_timeout_ms`` and ``region`` will overwrite settings in the configuration file or
           environment variables.
        4. The lack of AWS proper configuration can lead empty response. For more details related to S3 IO DataPipe
           setup and AWS config, please see the `README file`_.

    .. _README file:
        https://github.com/pytorch/data/tree/main/torchdata/datapipes/iter/load#s3-io-datapipe-documentation

    Args:
        source_datapipe: a DataPipe that contains URLs/URL prefixes to s3 files
        length: Nominal length of the datapipe
        request_timeout_ms: timeout setting for each reqeust (3,000ms by default)
        region: region for access files (inferred from credentials by default)

    Example:

    .. testsetup::

        from unittest import mock
        from torchdata.datapipes.iter import IterableWrapper, S3FileLister

        file_lister_patch = mock.patch.object(S3FileLister, "__iter__", return_value=iter([]))
        file_lister_patch.start()

    .. testcode::

        from torchdata.datapipes.iter import IterableWrapper, S3FileLister

        s3_prefixes = IterableWrapper(['s3://bucket-name/folder/', ...])

        dp_s3_urls = S3FileLister(s3_prefixes)
        for d in dp_s3_urls:
            pass

        # Functional API
        dp_s3_urls = s3_prefixes.list_files_by_s3(request_timeout_ms=100)
        for d in dp_s3_urls:
            pass

    .. testcleanup::

        file_lister_patch.stop()
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[str],
        length: int = -1,
        request_timeout_ms=-1,
        region="",
        masks: Union[str, List[str]] = "",
    ) -> None:
        if not hasattr(torchdata, "_torchdata") or not hasattr(torchdata._torchdata, "S3Handler"):
            raise ModuleNotFoundError("TorchData must be built with BUILD_S3=1 to use this datapipe.")

        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.length: int = length
        self.handler = torchdata._torchdata.S3Handler(request_timeout_ms, region)
        self.masks = masks

    def __iter__(self) -> Iterator[str]:
        for prefix in self.source_datapipe:
            while True:
                urls = self.handler.list_files(prefix)
                for url in urls:
                    if match_masks(url, self.masks):
                        yield url
                if not urls:
                    break
            self.handler.clear_marker()

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length


@functional_datapipe("load_files_by_s3")
class S3FileLoaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Iterable DataPipe that loads Amazon S3 files from the given S3 URLs (functional name: ``load_files_by_s3``).
    ``S3FileLoader`` iterates all given S3 URLs in ``BytesIO`` format with ``(url, BytesIO)`` tuples.

    Note:
        1. ``source_datapipe`` **must** contain a list of valid S3 URLs.
        2. ``request_timeout_ms`` and ``region`` will overwrite settings in the
           configuration file or environment variables.
        3. The lack of AWS proper configuration can lead empty response. For more details related to S3 IO DataPipe
           setup and AWS config, please see the `README file`_.

    .. _README file:
        https://github.com/pytorch/data/tree/main/torchdata/datapipes/iter/load#s3-io-datapipe-documentation

    Args:
        source_datapipe: a DataPipe that contains URLs to s3 files
        request_timeout_ms: timeout setting for each reqeust (3,000ms by default)
        region: region for access files (inferred from credentials by default)
        buffer_size: buffer size of each chunk to download large files progressively (128Mb by default)
        multi_part_download: flag to split each chunk into small packets and download those packets in parallel (enabled by default)

    Example:

    .. testsetup::

        from unittest import mock
        from torchdata.datapipes.iter import S3FileLister

        file_lister_patch = mock.patch.object(S3FileLister, "__iter__", return_value=iter([]))
        file_lister_patch.start()

    .. testcode::

        from torchdata.datapipes.iter import IterableWrapper, S3FileLoader

        dp_s3_urls = IterableWrapper(['s3://bucket-name/folder/', ...]).list_files_by_s3()
        # In order to make sure data are shuffled and sharded in the
        # distributed environment, `shuffle`  and `sharding_filter`
        # are required. For detail, please check our tutorial in:
        # https://pytorch.org/data/main/tutorial.html#working-with-dataloader
        sharded_s3_urls = dp_s3_urls.shuffle().sharding_filter()

        dp_s3_files = S3FileLoader(sharded_s3_urls)
        for url, fd in dp_s3_files: # Start loading data
            data = fd.read()

        # Functional API
        dp_s3_files = sharded_s3_urls.load_files_by_s3(buffer_size=256)
        for url, fd in dp_s3_files:
            data = fd.read()

    .. testcleanup::

        file_lister_patch.stop()
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[str],
        request_timeout_ms=-1,
        region="",
        buffer_size=None,
        multi_part_download=None,
    ) -> None:
        if not hasattr(torchdata, "_torchdata") or not hasattr(torchdata._torchdata, "S3Handler"):
            raise ModuleNotFoundError("TorchData must be built with BUILD_S3=1 to use this datapipe.")

        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.handler = torchdata._torchdata.S3Handler(request_timeout_ms, region)
        if buffer_size:
            self.handler.set_buffer_size(buffer_size)
        if multi_part_download:
            self.handler.set_multi_part_download(multi_part_download)

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            yield url, StreamWrapper(BytesIO(self.handler.s3_read(url)))

    def __len__(self) -> int:
        return len(self.source_datapipe)
