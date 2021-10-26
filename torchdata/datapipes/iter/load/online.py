# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Iterator, Tuple
from urllib.parse import urlparse
from requests.exceptions import HTTPError, RequestException
import re

import requests

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


def _get_response_from_http(url: str, *, timeout: float) -> Tuple[str, StreamWrapper]:
    try:
        with requests.Session() as session:
            if timeout is None:
                r = session.get(url, stream=True)
            else:
                r = session.get(url, timeout=timeout, stream=True)
        return url, StreamWrapper(r.raw)
    except HTTPError as e:
        raise Exception(f"Could not get the file. [HTTP Error] {e.response}.")
    except RequestException as e:
        raise Exception(f"Could not get the file at {url}. [RequestException] {e.response}.")
    except Exception:
        raise


class HTTPReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r""":class:`HTTPReaderIterDataPipe`

    Iterable DataPipe that takes file URLs (http URLs pointing to files), and
    yields tuples of file URL and IO stream

    Args:
        source_datapipe: a DataPipe that contains URLs
        timeout : timeout in seconds for http request
    """

    def __init__(self, source_datapipe: IterDataPipe[str], timeout=None) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.timeout = timeout

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            yield _get_response_from_http(url, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.source_datapipe)


def _get_response_from_google_drive(url: str) -> Tuple[str, StreamWrapper]:
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v
    if confirm_token is None:
        if "Quota exceeded" in str(response.content):
            raise RuntimeError(
                "Google drive link {} is currently unavailable, because the quota was exceeded.".format(url)
            )

    if confirm_token:
        url = url + "&confirm=" + confirm_token

    response = session.get(url, stream=True)

    if "content-disposition" not in response.headers:
        raise RuntimeError("Internal error: headers don't contain content-disposition.")

    filename = re.findall('filename="(.+)"', response.headers["content-disposition"])
    if filename is None:
        raise RuntimeError("Filename could not be autodetected")
    return filename[0], StreamWrapper(response.raw)


class GDriveReaderDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Iterable DataPipe that takes URLs point at GDrive files, and
    yields tuples of file name and IO stream

    Args:
        source_datapipe: a DataPipe that contains URLs to GDrive files
    """

    def __init__(self, source_datapipe: IterDataPipe[str]) -> None:
        self.source_datapip: IterDataPipe[str] = source_datapipe

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            yield _get_response_from_google_drive(url)

    def __len__(self) -> int:
        return len(self.source_datapipe)


class OnlineReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r""":class:
    Iterable DataPipe that takes file URLs (can be http URLs pointing to files or URLs to GDrive files), and
    yields tuples of file URL and IO stream

    Args:
        source_datapipe: a DataPipe that contains URLs
        timeout : timeout in seconds for http request
    """

    def __init__(self, source_datapipe: IterDataPipe[str], *, timeout=None) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.timeout = timeout

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            parts = urlparse(url)

            if re.match(r"(drive|docs)[.]google[.]com", parts.netloc):
                # TODO: can this also have a timeout?
                yield _get_response_from_google_drive(url)
            else:
                yield _get_response_from_http(url, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.source_datapipe)
