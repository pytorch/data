# Copyright (c) Facebook, Inc. and its affiliates.
import requests
import re
from torch.utils.data import IterDataPipe


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
                "Google drive link {} is currently unavailable, because the quota was exceeded.".format(
                    url
                ))

    if confirm_token:
        url = url + "&confirm=" + confirm_token

    response = session.get(url, stream=True)

    if 'content-disposition' not in response.headers:
        raise RuntimeError(
            "Internal error: headers don't contain content-disposition.")

    filename = re.findall("filename=\"(.+)\"",
                          response.headers['content-disposition'])
    if filename is None:
        raise RuntimeError("Filename could not be autodetected")
    filename = filename[0]

    return filename, response.raw


class GDriveReaderDataPipe(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for url in self.source_datapipe:
            yield _get_response_from_google_drive(url)

    def __len__(self):
        return len(self.source_datapipe)
