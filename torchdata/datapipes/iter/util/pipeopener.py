# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import subprocess
import sys
import urllib.parse
from typing import Dict, Iterator, List, Union

from torch.utils.data.datapipes.utils.common import StreamWrapper

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


if os.name == "nt":
    default_popen_methods = {
        "file": ["cat", "{path}"],
        "http": ["c:\\Windows\\System32\\curl", "-s", "-L", "{url}"],
        "https": ["c:\\Windows\\System32\\curl", "-s", "-L", "{url}"],
        "gs": ["gsutil", "cat", "{url}"],
        "s3": ["aws", "s3", "{url}", "-"],
        "ais": ["ais", "cat", "{url}"],
    }
else:
    default_popen_methods = {
        "file": ["cat", "{path}"],
        "http": ["curl", "-s", "-L", "{url}"],
        "https": ["curl", "-s", "-L", "{url}"],
        "gs": ["gsutil", "cat", "{url}"],
        "s3": ["aws", "s3", "{url}", "-"],
        "ais": ["ais", "cat", "{url}"],
    }


def _re_search(regex, s, group=0, default=""):
    if s is None:
        return default
    m = re.search(regex, s)
    if m:
        return m.group(group)
    return default


@functional_datapipe("popen")
class PipeOpenerIterDataPipe(IterDataPipe[Dict]):
    r"""
    Given a stream of urls, open those urls and returns a stream of `(url, stream)` pairs.

    This uses subprocesses the open URLs. The use of subprocesses means that I/O can be
    asynchronous and that any kind of command line tool can be used for accessing
    remote servers.

    URL schemes are mapped to commands by specifying keyword arguments. Default command
    lines are provided for opening `file`, `http`, `https`, `gs`, `s3`, and `ais`.

    Command lines can be specified either as a single string, passed to a shell,
    or as a list. Either way, url components can be referenced in the command line
    using url, path, query, params, fragment, netloc, scheme, dirname, topdir, fname.

    The `pipe:` scheme can be used for specifying arbitrary commands as inputs.

    Args:
        source_datapipe: a DataPipe yielding a stream of pairs, as returned by `load_from_tar`
        verbose: print command lines before executing
        **methods: `scheme=command_line` pairs

    Returns:
        a DataPipe yielding a stream of `(fname, data)` pairs

    Examples:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>>
        >>> dp = (
        >>>     IterableWrapper(["http://google.com", "http://facebook.com", "pipe:echo hello"])
        >>>     # override default http opener
        >>>     .popen(http=["lynx", "-dump", "{url}"])
        >>> )
        >>> for url, text in dp:
        >>>     print(url, repr(text)[:40])
    """

    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]], verbose=False, **methods) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe
        self.methods = dict(default_popen_methods)
        self.methods.update(methods)
        self.verbose = verbose

    def __iter__(self) -> Iterator[Dict]:
        for url in self.source_datapipe:
            if not isinstance(url, str):
                raise TypeError(f"Expected string type for url, but got {type(url)}.")
            if url.lower().startswith("pipe:"):
                cmd = url[5:]
                if self.verbose:
                    print(f"# {cmd}", file=sys.stderr)
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            else:
                o = urllib.parse.urlparse(url)
                scheme = o.scheme or "file"
                handler = self.methods.get(scheme.lower())
                if handler is None:
                    raise ValueError(f"No known popen handler for '{o.scheme}' ({url[:60]}).")
                kw = dict(
                    url=url,
                    path=o.path,
                    query=o.query,
                    params=o.params,
                    fragment=o.fragment,
                    netloc=o.netloc,
                    scheme=o.scheme,
                    dirname=_re_search("^(.*)/", o.path, group=1),
                    topdir=_re_search("^(.*?)/", o.path, group=1),
                    fname=_re_search("^.*/(.*?)$", o.path, group=1),
                )
                if isinstance(handler, list):
                    cmd = [x.format(**kw) for x in handler]
                    if self.verbose:
                        print(f"# {cmd}", file=sys.stderr)
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                else:
                    cmd = handler.format(**kw)
                    if self.verbose:
                        print(f"# {cmd}", file=sys.stderr)
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            yield url, StreamWrapper(proc.stdout)

    def __len__(self) -> int:
        return len(self.source_datapipe)
