import urllib.request as urllib
from io import IOBase
from typing import Tuple
from urllib.error import HTTPError, URLError

from torch.utils.data import IterDataPipe

# TODO(VitalyFedyunin): This file copy-pasted from the pytorch repo
# nuke torch/utils/data/datapipes/iter/httpreader.py, when torchdata
# is open sourced

class HTTPReaderIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    r""" :class:`HTTPReaderIterDataPipe`

    Iterable DataPipe to load file url(s) (http url(s) pointing to file(s)),
    yield file url and IO stream in a tuple
    args:
        timeout : timeout for http request
    """

    def __init__(self, source_datapipe, timeout=None):
        self.source_datapipe = source_datapipe
        self.timeout = timeout

    def __iter__(self):
        for furl in self.source_datapipe:
            try:
                if self.timeout is None:
                    r = urllib.urlopen(furl)
                else:
                    r = urllib.urlopen(furl, timeout=self.timeout)

                yield(furl, r)
            except HTTPError as e:
                raise Exception("Could not get the file.\
                                [HTTP Error] {code}: {reason}."
                                .format(code=e.code, reason=e.reason))
            except URLError as e:
                raise Exception("Could not get the file at {url}.\
                                 [URL Error] {reason}."
                                .format(reason=e.reason, url=furl))
            except Exception:
                raise
