# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import urllib

from typing import Dict, Iterator, Optional, Tuple

import requests

from requests.exceptions import HTTPError, RequestException

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


# TODO: Remove this helper function when https://bugs.python.org/issue42627 is resolved
def _get_proxies() -> Optional[Dict[str, str]]:
    import os

    if os.name == "nt":
        proxies = urllib.request.getproxies()
        address = proxies.get("https")
        # The default proxy type of Windows is HTTP
        if address and address.startswith("https"):
            address = "http" + address[5:]
            proxies["https"] = address
            return proxies
    return None


def _get_response_from_http(
    url: str, *, timeout: Optional[float], query_params: Optional[Dict[str, str]]
) -> Tuple[str, StreamWrapper]:
    try:
        with requests.Session() as session:
            proxies = _get_proxies()
            if timeout is None:
                r = session.get(url, stream=True, proxies=proxies, **query_params)
            else:
                r = session.get(url, timeout=timeout, stream=True, proxies=proxies, **query_params)
        return url, StreamWrapper(r.raw)
    except HTTPError as e:
        raise Exception(f"Could not get the file. [HTTP Error] {e.response}.")
    except RequestException as e:
        raise Exception(f"Could not get the file at {url}. [RequestException] {e.response}.")
    except Exception:
        raise


@functional_datapipe("read_from_http")
class HTTPReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes file URLs (HTTP URLs pointing to files), and yields tuples of file URL and
    IO stream (functional name: ``read_from_http``).

    Args:
        source_datapipe: a DataPipe that contains URLs
        timeout: timeout in seconds for HTTP request

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, HttpReader
        >>> file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        >>> http_reader_dp = HttpReader(IterableWrapper([file_url]))
        >>> reader_dp = http_reader_dp.readlines()
        >>> it = iter(reader_dp)
        >>> path, line = next(it)
        >>> path
        https://raw.githubusercontent.com/pytorch/data/main/LICENSE
        >>> line
        b'BSD 3-Clause License'
    """

    def __init__(self, source_datapipe: IterDataPipe[str], timeout: Optional[float] = None, **kwargs) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.timeout = timeout
        self.query_params = kwargs

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            yield _get_response_from_http(url, timeout=self.timeout, query_params=self.query_params)

    def __len__(self) -> int:
        return len(self.source_datapipe)


def _get_response_from_google_drive(url: str, *, timeout: Optional[float]) -> Tuple[str, StreamWrapper]:
    confirm_token = None

    with requests.Session() as session:
        if timeout is None:
            response = session.get(url, stream=True)
        else:
            response = session.get(url, timeout=timeout, stream=True)
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                confirm_token = v
        if confirm_token is None:
            if "Quota exceeded" in str(response.content):
                raise RuntimeError(f"Google drive link {url} is currently unavailable, because the quota was exceeded.")

        if confirm_token:
            url = url + "&confirm=" + confirm_token

        if timeout is None:
            response = session.get(url, stream=True)
        else:
            response = session.get(url, timeout=timeout, stream=True)

        if "content-disposition" not in response.headers:
            raise RuntimeError("Internal error: headers don't contain content-disposition.")

        filename = re.findall('filename="(.+)"', response.headers["content-disposition"])
        if filename is None:
            raise RuntimeError("Filename could not be autodetected")

    return filename[0], StreamWrapper(response.raw)


@functional_datapipe("read_from_gdrive")
class GDriveReaderDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes URLs pointing at GDrive files, and yields tuples of file name and
    IO stream (functional name: ``read_from_gdrive``).

    Args:
        source_datapipe: a DataPipe that contains URLs to GDrive files
        timeout: timeout in seconds for HTTP request

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, GDriveReader
        >>> gdrive_file_url = "https://drive.google.com/uc?export=download&id=SomeIDToAGDriveFile"
        >>> gdrive_reader_dp = GDriveReader(IterableWrapper([gdrive_file_url]))
        >>> reader_dp = gdrive_reader_dp.readlines()
        >>> it = iter(reader_dp)
        >>> path, line = next(it)
        >>> path
        https://drive.google.com/uc?export=download&id=SomeIDToAGDriveFile
        >>> line
        <First line from the GDrive File>
    """
    source_datapipe: IterDataPipe[str]

    def __init__(self, source_datapipe: IterDataPipe[str], *, timeout: Optional[float] = None) -> None:
        self.source_datapipe = source_datapipe
        self.timeout = timeout

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            yield _get_response_from_google_drive(url, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.source_datapipe)


@functional_datapipe("read_from_remote")
class OnlineReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes file URLs (can be HTTP URLs pointing to files or URLs to GDrive files), and
    yields tuples of file URL and IO stream (functional name: ``read_from_remote``).

    Args:
        source_datapipe: a DataPipe that contains URLs
        timeout: timeout in seconds for HTTP request

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, OnlineReader
        >>> file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        >>> online_reader_dp = OnlineReader(IterableWrapper([file_url]))
        >>> reader_dp = online_reader_dp.readlines()
        >>> it = iter(reader_dp)
        >>> path, line = next(it)
        >>> path
        https://raw.githubusercontent.com/pytorch/data/main/LICENSE
        >>> line
        b'BSD 3-Clause License'
    """
    source_datapipe: IterDataPipe[str]

    def __init__(self, source_datapipe: IterDataPipe[str], *, timeout: Optional[float] = None) -> None:
        self.source_datapipe = source_datapipe
        self.timeout = timeout

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            parts = urllib.parse.urlparse(url)

            if re.match(r"(drive|docs)[.]google[.]com", parts.netloc):
                yield _get_response_from_google_drive(url, timeout=self.timeout)
            else:
                yield _get_response_from_http(url, timeout=self.timeout)

    def __len__(self) -> int:
        return len(self.source_datapipe)
