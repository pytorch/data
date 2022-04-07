from io import BytesIO
from typing import Iterator, Tuple

import torchdata
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("list_file_by_s3")
class S3FileListerIterDataPipe(IterDataPipe[str]):
    r""":class:`S3FileListerIterDataPipe`.

    Iterable DataPipe that lists URLs with the given prefixes (functional name: ``list_file_by_s3``).
    Acceptable prefixes include `s3://bucket-name`, `s3://bucket-name/`, `s3://bucket-name/folder`,
    `s3://bucket-name/folder/`, and `s3://bucket-name/prefix`. You may also set `length`, `request_timeout_ms` (default 3000
    ms in aws-sdk-cpp), and `region`. Note that:

    1. Input **must** be a list and direct S3 URLs are skipped.
    2. `length` is `-1` by default, and any call to `__len__()` is invalid, because the length is unknown until all files
    are iterated.
    3. `request_timeout_ms` and `region` will overwrite settings in the configuration file or environment variables.

    Args:
        source_datapipe: a DataPipe that contains URLs/URL prefixes to s3 files
        length: Nominal length of the datapipe
        requestTimeoutMs: optional, overwrite the default timeout setting for this datapipe
        region: optional, overwrite the default region inferred from credentials for this datapipe

    Note:
        AWS_CPP_SDK is necessary to use the S3 DataPipe(s).

    Example:
        >>> from torchdata.datapipes.iter import S3FileLister, S3FileLoader
        >>> s3_prefixes = ['s3://bucket-name/folder/', ...]
        >>> dp_s3_urls = S3FileLister(s3_prefixes)
        >>> dp_s3_files = S3FileLoader(s3_urls) # outputs in (url, StreamWrapper(BytesIO))
        >>> # more datapipes to convert loaded bytes, e.g.
        >>> datapipe = dp_s3_files.parse_csv(delimiter=' ')
        >>> for d in datapipe: # Start loading data
        ...     pass
    """

    def __init__(self, source_datapipe: IterDataPipe[str], length: int = -1, request_timeout_ms=-1, region="") -> None:
        if not hasattr(torchdata, "_torchdata") or not hasattr(torchdata._torchdata, "S3Handler"):
            raise ModuleNotFoundError("Torchdata must be built with BUILD_S3=1 to use this datapipe.")

        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.length: int = length
        self.handler = torchdata._torchdata.S3Handler(request_timeout_ms, region)

    def __iter__(self) -> Iterator[str]:
        for prefix in self.source_datapipe:
            while True:
                urls = self.handler.list_files(prefix)
                yield from urls
                if not urls:
                    break
            self.handler.clear_marker()

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length


@functional_datapipe("load_file_by_s3")
class S3FileLoaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r""":class:`S3FileListerIterDataPipe`.

    Iterable DataPipe that loads S3 files given S3 URLs (functional name: ``load_file_by_s3``).
    `S3FileLoader` iterates all given S3 URLs in `BytesIO` format with `(url, BytesIO)` tuples.
    You may also set `request_timeout_ms` (default 3000 ms in aws-sdk-cpp), `region`,
    `buffer_size` (default 120Mb), and `multi_part_download` (default to use multi-part downloading). Note that:

    1. Input **must** be a list and S3 URLs must be valid.
    2. `request_timeout_ms` and `region` will overwrite settings in the configuration file or environment variables.

    Args:
        source_datapipe: a DataPipe that contains URLs to s3 files
        requestTimeoutMs: optional, overwrite the default timeout setting for this datapipe
        region: optional, overwrite the default region inferred from credentials for this datapipe

    Note:
        AWS_CPP_SDK is necessary to use the S3 DataPipe(s).

    Example:
        >>> from torchdata.datapipes.iter import S3FileLister, S3FileLoader
        >>> s3_prefixes = ['s3://bucket-name/folder/', ...]
        >>> dp_s3_urls = S3FileLister(s3_prefixes)
        >>> dp_s3_files = S3FileLoader(s3_urls) # outputs in (url, StreamWrapper(BytesIO))
        >>> # more datapipes to convert loaded bytes, e.g.
        >>> datapipe = dp_s3_files.parse_csv(delimiter=' ')
        >>> for d in datapipe: # Start loading data
        ...     pass
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
            raise ModuleNotFoundError("Torchdata must be built with BUILD_S3=1 to use this datapipe.")

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
