# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import urllib
import warnings
from typing import Any, Dict, Iterator, Optional, Tuple

import requests

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


# TODO(642): Remove this helper function when https://bugs.python.org/issue42627 is resolved
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
    url: str, *, timeout: Optional[float], **query_params: Optional[Dict[str, Any]]
) -> Tuple[str, StreamWrapper]:
    with requests.Session() as session:
        proxies = _get_proxies()
        r = session.get(url, timeout=timeout, proxies=proxies, stream=True, **query_params)  # type: ignore[attr-defined]
    r.raise_for_status()
    return url, StreamWrapper(r.raw)


@functional_datapipe("read_from_http")
class HTTPReaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes file URLs (HTTP URLs pointing to files), and yields tuples of file URL and
    IO stream (functional name: ``read_from_http``).

    Args:
        source_datapipe: a DataPipe that contains URLs
        timeout: timeout in seconds for HTTP request
        skip_on_error: whether to skip over urls causing problems, otherwise an exception is raised
        **kwargs: a Dictionary to pass optional arguments that requests takes. For the full list check out https://docs.python-requests.org/en/master/api/

    Example:

    .. testcode::

        from torchdata.datapipes.iter import IterableWrapper, HttpReader

        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        query_params = {"auth" : ("fake_username", "fake_password"), "allow_redirects" : True}
        timeout = 120
        http_reader_dp = HttpReader(IterableWrapper([file_url]), timeout=timeout, **query_params)
        reader_dp = http_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        print((path, line))

    Output:

    .. testoutput::

        ('https://raw.githubusercontent.com/pytorch/data/main/LICENSE', b'BSD 3-Clause License')
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[str],
        timeout: Optional[float] = None,
        skip_on_error: bool = False,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.timeout = timeout
        self.skip_on_error = skip_on_error
        self.query_params = kwargs

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            try:
                yield _get_response_from_http(url, timeout=self.timeout, **self.query_params)
            except Exception as e:
                if self.skip_on_error:
                    warnings.warn(f"{e}, skipping...")
                else:
                    raise

    def __len__(self) -> int:
        return len(self.source_datapipe)


def _extract_gdrive_api_response(content: str) -> Optional[str]:
    match = re.search("<title>Google Drive - (?P<api_response>.+?)</title>", content)
    return match["api_response"] if match is not None else None


def _get_response_from_google_drive(
    url: str, *, timeout: Optional[float], **query_params: Optional[Dict[str, Any]]
) -> Tuple[str, StreamWrapper]:
    confirm_token = None

    with requests.Session() as session:
        response = session.get(url, timeout=timeout, stream=True, **query_params)  # type: ignore[attr-defined]
        response.raise_for_status()

        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                confirm_token = v
                break
        else:
            api_response = _extract_gdrive_api_response(response.text)
            if api_response == "Virus scan warning":
                confirm_token = "t"
            elif api_response == "Quota exceeded":
                raise RuntimeError(f"Google drive link {url} is currently unavailable, because the quota was exceeded.")

        if confirm_token:
            url = url + "&confirm=" + confirm_token

        response = session.get(url, timeout=timeout, stream=True, **query_params)  # type: ignore[attr-defined]
        response.raise_for_status()

        if "content-disposition" not in response.headers:
            raise RuntimeError(
                f"Google drive link {url} internal error: "
                "headers don't contain content-disposition. This is usually caused by "
                "using a sharing/viewing link instead of a download link. Click 'Download' on the "
                "Google Drive page, which should redirect you to a download page, and use the link "
                "of that page."
            )

        filename = re.findall('filename="(.+)"', response.headers["content-disposition"])
        if filename is None:
            raise RuntimeError(f"Google drive link {url}: filename could not be autodetected")

    return filename[0], StreamWrapper(response.raw)


@functional_datapipe("read_from_gdrive")
class GDriveReaderDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Takes URLs pointing at GDrive files, and yields tuples of file name and
    IO stream (functional name: ``read_from_gdrive``).

    Args:
        source_datapipe: a DataPipe that contains URLs to GDrive files
        timeout: timeout in seconds for HTTP request
        skip_on_error: whether to skip over urls causing problems, otherwise an exception is raised
        **kwargs: a Dictionary to pass optional arguments that requests takes. For the full list check out https://docs.python-requests.org/en/master/api/

    Example:

    .. testsetup::

        from torchdata.datapipes.iter import GDriveReader

        GDriveReader.readlines = lambda self: [
            ("https://drive.google.com/uc?export=download&id=SomeIDToAGDriveFile", b"<First line from the GDrive File>")
        ]

    .. testcode::

        from torchdata.datapipes.iter import IterableWrapper, GDriveReader

        gdrive_file_url = "https://drive.google.com/uc?export=download&id=SomeIDToAGDriveFile"
        gdrive_reader_dp = GDriveReader(IterableWrapper([gdrive_file_url]))
        reader_dp = gdrive_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        print((path, line))

    Output:

    .. testoutput::

        ('https://drive.google.com/uc?export=download&id=SomeIDToAGDriveFile', b'<First line from the GDrive File>')
    """
    source_datapipe: IterDataPipe[str]

    def __init__(
        self,
        source_datapipe: IterDataPipe[str],
        *,
        timeout: Optional[float] = None,
        skip_on_error: bool = False,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        self.source_datapipe = source_datapipe
        self.timeout = timeout
        self.skip_on_error = skip_on_error
        self.query_params = kwargs

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            try:
                yield _get_response_from_google_drive(url, timeout=self.timeout, **self.query_params)
            except Exception as e:
                if self.skip_on_error:
                    warnings.warn(f"{e}, skipping...")
                else:
                    raise

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
        skip_on_error: whether to skip over urls causing problems, otherwise an exception is raised
        **kwargs: a Dictionary to pass optional arguments that requests takes. For the full list check out https://docs.python-requests.org/en/master/api/

    Example:

    .. testcode::

        from torchdata.datapipes.iter import IterableWrapper, OnlineReader

        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        online_reader_dp = OnlineReader(IterableWrapper([file_url]))
        reader_dp = online_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        print((path, line))

    Output:

    .. testoutput::

        ('https://raw.githubusercontent.com/pytorch/data/main/LICENSE', b'BSD 3-Clause License')
    """
    source_datapipe: IterDataPipe[str]

    def __init__(
        self,
        source_datapipe: IterDataPipe[str],
        *,
        timeout: Optional[float] = None,
        skip_on_error: bool = False,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        self.source_datapipe = source_datapipe
        self.timeout = timeout
        self.skip_on_error = skip_on_error
        self.query_params = kwargs

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            parts = urllib.parse.urlparse(url)

            if re.match(r"(drive|docs)[.]google[.]com", parts.netloc):
                try:
                    yield _get_response_from_google_drive(url, timeout=self.timeout, **self.query_params)
                except Exception as e:
                    if self.skip_on_error:
                        warnings.warn(f"{e}, skipping...")
                    else:
                        raise
            else:
                try:
                    yield _get_response_from_http(url, timeout=self.timeout, **self.query_params)
                except Exception as e:
                    if self.skip_on_error:
                        warnings.warn(f"{e}, skipping...")
                    else:
                        raise

    def __len__(self) -> int:
        return len(self.source_datapipe)
