from io import BytesIO
from typing import Iterator, Tuple

import torchdata
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


@functional_datapipe("list_file_by_s3")
class S3FileListerIterDataPipe(IterDataPipe[str]):
    r""":class:`S3FileListerIterDataPipe`.

    Iterable DataPipe that lists URLs with the given prefixes.

    Args:
        source_datapipe: a DataPipe that contains URLs/URL prefixes to s3 files
        length: Nominal length of the datapipe
        requestTimeoutMs: optional, overwrite the default timeout setting for this datapipe
        region: optional, overwrite the default region inferred from credentials for this datapipe

    Note:
        AWS_CPP_SDK is necessary to use the S3 DataPipe(s).
    """

    def __init__(self, source_datapipe: IterDataPipe[str], length: int = -1, request_timeout_ms=-1, region="") -> None:
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

    Iterable DataPipe that loads S3 files given S3 URLs.

    Args:
        source_datapipe: a DataPipe that contains URLs to s3 files
        requestTimeoutMs: optional, overwrite the default timeout setting for this datapipe
        region: optional, overwrite the default region inferred from credentials for this datapipe

    Note:
        AWS_CPP_SDK is necessary to use the S3 DataPipe(s).
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[str],
        request_timeout_ms=-1,
        region="",
        buffer_size=None,
        multi_part_download=None,
    ) -> None:
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
