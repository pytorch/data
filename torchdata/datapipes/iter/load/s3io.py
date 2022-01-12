from typing import Iterator, Tuple
from itertools import chain

# import torchdata._torchdata
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

    Note:
        AWS_CPP_SDK is necessary to use the S3 DataPipe(s).
    """

    def __init__(self, source_datapipe: IterDataPipe[str], length: int = -1) -> None:
        # TODO: check whether AWS_CPP_SDK available
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[str]:
        # TODO: timeout
        handler = torchdata._torchdata.S3Handler()
        for prefix in self.source_datapipe:
            # if handler.file_exists(prefix):
            #     yield prefix
            # else:
            #     urls = handler.list_files(prefix)
            #     for url in urls:
            #         yield url
            urls = handler.list_files(prefix)
            for url in urls:
                yield url

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length


@functional_datapipe("load_file_by_s3")
class S3FileLoaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r""":class:`S3FileListerIterDataPipe`.

    Iterable DataPipe that loads S3 files given S3 URLs.

    Args:
        source_datapipe: a DataPipe that contains URLs to s3 files

    Note:
        AWS_CPP_SDK is necessary to use the S3 DataPipe(s).
    """

    def __init__(self, source_datapipe: IterDataPipe[str]) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        # TODO: timeout
        handler = torchdata._torchdata.S3Handler()
        for url in self.source_datapipe:
            yield url, handler.s3_read(url)

    def __len__(self) -> int:
        return len(self.source_datapipe)
