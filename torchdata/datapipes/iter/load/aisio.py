# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import Iterator, Tuple

from torchdata.datapipes import functional_datapipe

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

try:
    from aistore.client import Client
    from aistore.pytorch.utils import parse_url, unparse_url

    HAS_AIS = True
except ImportError:
    HAS_AIS = False


def _assert_aistore() -> None:
    if not HAS_AIS:
        raise ModuleNotFoundError(
            "Package `aistore` is required to be installed to use this datapipe."
            "Please run `pip install aistore` or `conda install aistore` to install the package"
            "For more info visit: https://github.com/NVIDIA/aistore/blob/master/sdk/python/"
        )


@functional_datapipe("list_files_by_ais")
class AISFileListerIterDataPipe(IterDataPipe[str]):
    """
    Iterable Datapipe that lists files from the AIStore backends with the given URL prefixes. (functional name: ``list_files_by_ais``).
    Acceptable prefixes include but not limited to - `ais://bucket-name`, `ais://bucket-name/`

    Note:
    -   This function also supports files from multiple backends (`aws://..`, `gcp://..`, `azure://..`, etc)
    -   Input must be a list and direct URLs are not supported.
    -   length is -1 by default, all calls to len() are invalid as
        not all items are iterated at the start.
    -   This internally uses AIStore Python SDK.

    Args:
        source_datapipe(IterDataPipe[str]): a DataPipe that contains URLs/URL
                                            prefixes to objects on AIS
        url(str): AIStore endpoint
        length(int): length of the datapipe

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, AISFileLister
        >>> ais_prefixes = IterableWrapper(['gcp://bucket-name/folder/', 'aws:bucket-name/folder/', 'ais://bucket-name/folder/', ...])
        >>> dp_ais_urls = AISFileLister(url='localhost:8080', source_datapipe=prefix)
        >>> for url in dp_ais_urls:
        ...     pass
        >>> # Functional API
        >>> dp_ais_urls = ais_prefixes.list_files_by_ais(url='localhost:8080')
        >>> for url in dp_ais_urls:
        ...     pass
    """

    def __init__(self, source_datapipe: IterDataPipe[str], url: str, length: int = -1) -> None:
        _assert_aistore()
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.length: int = length
        self.client = Client(url)

    def __iter__(self) -> Iterator[str]:
        for prefix in self.source_datapipe:
            provider, bck_name, prefix = parse_url(prefix)
            obj_iter = self.client.list_objects_iter(bck_name=bck_name, provider=provider, prefix=prefix)
            for entry in obj_iter:
                yield unparse_url(provider=provider, bck_name=bck_name, obj_name=entry.name)

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length


@functional_datapipe("load_files_by_ais")
class AISFileLoaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    """
    Iterable DataPipe that loads files from AIStore with the given URLs (functional name: ``load_files_by_ais``).
    Iterates all files in BytesIO format and returns a tuple (url, BytesIO).

    Note:
    -   This function also supports files from multiple backends (`aws://..`, `gcp://..`, `azure://..`, etc)
    -   Input must be a list and direct URLs are not supported.
    -   This internally uses AIStore Python SDK.

    Args:
        source_datapipe(IterDataPipe[str]): a DataPipe that contains URLs/URL prefixes to objects
        url(str): AIStore endpoint
        length(int): length of the datapipe

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, AISFileLister,AISFileLoader
        >>> ais_prefixes = IterableWrapper(['gcp://bucket-name/folder/', 'aws:bucket-name/folder/', 'ais://bucket-name/folder/', ...])
        >>> dp_ais_urls = AISFileLister(url='localhost:8080', source_datapipe=prefix)
        >>> dp_cloud_files = AISFileLoader(url='localhost:8080', source_datapipe=dp_ais_urls)
        >>> for url, file in dp_cloud_files:
        ...     pass
        >>> # Functional API
        >>> dp_cloud_files = dp_ais_urls.load_files_by_ais(url='localhost:8080')
        >>> for url, file in dp_cloud_files:
        ...     pass
    """

    def __init__(self, source_datapipe: IterDataPipe[str], url: str, length: int = -1) -> None:
        _assert_aistore()
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.length = length
        self.client = Client(url)

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            provider, bck_name, obj_name = parse_url(url)
            yield url, StreamWrapper(
                BytesIO(self.client.get_object(bck_name=bck_name, provider=provider, obj_name=obj_name).read_all())
            )

    def __len__(self) -> int:
        return len(self.source_datapipe)
