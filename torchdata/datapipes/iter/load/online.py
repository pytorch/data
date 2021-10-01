# Copyright (c) Facebook, Inc. and its affiliates.
import urllib.request as urllib
from io import IOBase
from typing import Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
import re

import requests

from torchdata.datapipes.iter import IterDataPipe

# TODO(VitalyFedyunin): This HTTP part is copy-pasted from the pytorch repo
# nuke torch/utils/data/datapipes/iter/httpreader.py, when torchdata
# is open sourced


def _get_response_from_http(url, *, timeout):
    try:
        if timeout is None:
            r = urllib.urlopen(url)
        else:
            r = urllib.urlopen(url, timeout=timeout)

        return (url, r)
    except HTTPError as e:
        raise Exception(
            "Could not get the file.\
                        [HTTP Error] {code}: {reason}.".format(
                code=e.code, reason=e.reason
            )
        )
    except URLError as e:
        raise Exception(
            "Could not get the file at {url}.\
                         [URL Error] {reason}.".format(
                reason=e.reason, url=url
            )
        )
    except Exception:
        raise


class HTTPReaderIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    r""":class:`HTTPReaderIterDataPipe`

    Iterable DataPipe that takes file URLs (http URLs pointing to files), and
    yields tuples of file URL and IO stream

    Args:
        source_datapipe: a DataPipe that contains URLs
        timeout : timeout in seconds for http request
    """

    def __init__(self, source_datapipe, timeout=None):
        self.source_datapipe = source_datapipe
        self.timeout = timeout

    def __iter__(self):
        for url in self.source_datapipe:
            yield _get_response_from_http(url, timeout=self.timeout)

    def __len__(self):
        return len(self.source_datapipe)


def _get_response_from_google_drive(url):
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
    filename = filename[0]

    return filename, response.raw


class GDriveReaderDataPipe(IterDataPipe):
    r"""
    Iterable DataPipe that takes URLs point at GDrive files, and
    yields tuples of file name and IO stream

    Args:
        source_datapipe: a DataPipe that contains URLs to GDrive files
    """

    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for url in self.source_datapipe:
            yield _get_response_from_google_drive(url)

    def __len__(self):
        return len(self.source_datapipe)


class OnlineReaderIterDataPipe(IterDataPipe):
    r""":class:
    Iterable DataPipe that takes file URLs (can be http URLs pointing to files or URLs to GDrive files), and
    yields tuples of file URL and IO stream

    Args:
        source_datapipe: a DataPipe that contains URLs
        timeout : timeout in seconds for http request
    """

    def __init__(self, source_datapipe, *, timeout=None):
        self.source_datapipe = source_datapipe
        self.timeout = timeout

    def __iter__(self):
        for url in self.source_datapipe:
            parts = urlparse(url)

            if re.match(r"(drive|docs)[.]google[.]com", parts.netloc):
                # TODO: can this also have a timeout?
                yield _get_response_from_google_drive(url)
            else:
                yield _get_response_from_http(url, timeout=self.timeout)

    def __len__(self):
        return len(self.source_datapipe)
