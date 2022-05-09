# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import re
import pickle
import subprocess
from typing import Any, Dict, Iterator, List, Union
from urllib.parse import urlparse
from fnmatch import fnmatch

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper


def pathsplit(p):
    """Split a path into a WebDataset prefix and suffix.

    The prefix is used for grouping files into samples,
    the suffix is used as key in the output dictionary.
    The suffix consists of all components after the last
    "." in the filename.

    In torchdata, the prefix consists of the .tar file
    path followed by the file name inside the archive.

    Any backslash in the prefix is replaced by a forward
    slash to make Windows prefixes consistent with POSIX
    paths.
    """

    # convert Windows pathnames to UNIX pathnames, otherwise
    # we get an inconsistent mix of the Windows path to the tar
    # file followed by the POSIX path inside that tar file
    p = p.replace("\\", "/")
    if "." not in p:
        return p, ""
    # we need to use a regular expression because os.path is
    # platform specific, but tar files always contain POSIX paths
    match = re.search(r"^(.*?)(\.[^/]*)$", p)
    if not match:
        return p, ""
    prefix, suffix = match.groups()
    return prefix, suffix


@functional_datapipe("webdataset")
class WebDatasetIterDataPipe(IterDataPipe[Dict]):
    r"""
    Iterable DataPipe that accepts stream of (path, data) tuples, usually,
    representing the pathnames and files of a tar archive (functional name:
    ``webdataset``). This aggregates consecutive items with the same basename
    into a single dictionary, using the extensions as keys (WebDataset file
    convention). Any text after the first "." in the filename is used as
    a key/extension.

    File names that do not have an extension are ignored.

    Args:
        source_datapipe: a DataPipe yielding a stream of (path, data) pairs

    Returns:
        a DataPipe yielding a stream of dictionaries

    Examples:
        >>> from torchdata.datapipes.iter import FileLister, FileOpener
        >>>
        >>> def decode(item):
        >>>     key, value = item
        >>>     if key.endswith(".txt"):
        >>>         return key, value.read().decode("utf-8")
        >>>     if key.endswith(".bin"):
        >>>         return key, value.read().decode("utf-8")
        >>>
        >>> datapipe1 = FileLister("test/_fakedata", "wds*.tar")
        >>> datapipe2 = FileOpener(datapipe1, mode="b")
        >>> dataset = datapipe2.load_from_tar().map(decode).webdataset()
        >>> for obj in dataset:
        >>>     print(obj)
    """

    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]]) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe

    def __iter__(self) -> Iterator[Dict]:
        sample: Dict[str, Any] = {}
        current = ""
        for path, data in self.source_datapipe:
            assert isinstance(path, str), path
            prefix, suffix = pathsplit(path)
            if suffix == "":
                # files with empty suffixes can be used for metadata
                # they cannot be used for data since they wouldn't have a key
                continue
            if prefix != current:
                if current != "":
                    yield sample
                sample = {}
                current = prefix
                sample["__key__"] = current
            sample[suffix] = data
        if sample != {}:
            yield sample

    def __len__(self) -> int:
        return len(self.source_datapipe)


def shardexpand(s):
    expansion = r"[{][0-9]+[.][.][0-9]+[}]"
    m = re.search(expansion, s)
    if not m:
        return [s]
    prefix = s[:m.start()]
    rest = shardexpand(s[m.end():])
    rng = s[m.start() + 1:m.end() - 1]
    lohi = rng.split("..")
    if len(lohi[0]) != len(lohi[1]):
        raise ValueError(
            "Shard specifications must have " +
            f"same number of digits for low and high values in {s}."
        )
    lo, hi = [int(x) for x in lohi]
    if lo >= hi:
        raise ValueError(f"Bad range in in shard spec {s}.")
    result = []
    for i in range(lo, hi + 1):
        for r in rest:
            expanded = f"{prefix}{i:0>{len(lohi[1])}}{r}"
            result.append(expanded)
    return result


@functional_datapipe("shardexpand")
class ShardExpanderIterDataPipe(IterDataPipe[Dict]):
    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]]) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe

    def __iter__(self) -> Iterator[Dict]:
        for path in self.source_datapipe:
            for expanded in shardexpand(path):
                yield expanded

    def __len__(self) -> int:
        return len(self.source_datapipe)


def decode_bin(stream):
    return stream.read()


def decode_text(stream):
    binary = stream.read()
    return binary.decode("utf-8")


def decode_pickle(stream):
    return pickle.load(stream)


default_decoders = [
    ("*.bin", decode_bin),
    ("*.txt", decode_text),
    ("*.pyd", decode_pickle),
]


def find_decoder(decoders, path):
    fname = re.sub(r".*/", "", path)
    if fname.startswith("__"):
        return lambda x: x
    for pattern, fun in decoders[::-1]:
        if fnmatch(fname.lower(), pattern):
            return fun
    return None


@functional_datapipe("decode")
class FileDecoderIterDataPipe(IterDataPipe[Dict]):
    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]], *args, must_decode=True, **kw) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe
        self.must_decode = must_decode
        self.decoders = list(default_decoders) + list(args)
        self.decoders += [("*." + k, v) for k, v in kw.items()]

    def __iter__(self) -> Iterator[Dict]:
        for path, stream in self.source_datapipe:
            decoder = find_decoder(self.decoders, path)
            if decoder is None:
                if self.must_decode:
                    raise ValueError(f"No decoder found for {path}.")
                else:
                    value = stream.read()
            else:
                value = decoder(stream)
            yield path, value

    def __len__(self) -> int:
        return len(self.source_datapipe)


if os.name == "nt":
    default_popen_methods = {
        "file": ["cat", "{path}"],
        "http": ["c:\\Windows\\System32\\curl", "-s", "-L", "{url}"],
        "https": ["c:\\Windows\\System32\\curl", "-s", "-L", "{url}"],
        "gs": ["gsutil", "cat", "{url}"],
    }
else:
    default_popen_methods = {
        "file": ["cat", "{path}"],
        "http": ["curl", "-s", "-L", "{url}"],
        "https": ["curl", "-s", "-L", "{url}"],
        "gs": ["gsutil", "cat", "{url}"],
    }


@functional_datapipe("popen")
class PipeOpenerIterDataPipe(IterDataPipe[Dict]):
    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]], verbose=False, **methods) -> None:
        global default_popen_methods
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
                o = urlparse(url)
                scheme = o.scheme or "file"
                handler = self.methods.get(scheme)
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


@functional_datapipe("rename_keys")
class RenameKeysIterDataPipe(IterDataPipe[Dict]):
    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]], *args, keep_unselected=False, must_match=True, duplicate_is_error=True, **kw) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe
        self.must_match = must_match
        self.keep_unselected = keep_unselected
        self.duplicate_is_error = duplicate_is_error
        self.renamings = list(args) + [(v, k) for k, v in kw.items()]

    def __iter__(self) -> Iterator[Dict]:
        for sample in self.source_datapipe:
            new_sample = {}
            matched = {k: False for k, _ in self.renamings}
            for path, value in sample.items():
                fname = re.sub(r".*/", "", path)
                new_name = None
                for pattern, name in self.renamings[::-1]:
                    if fnmatch(fname.lower(), pattern):
                        matched[pattern] = True
                        new_name = name
                        break
                if new_name is None:
                    if self.keep_unselected:
                        new_sample[path] = value
                    continue
                if new_name in new_sample:
                    if self.duplicate_is_error:
                        raise ValueError(f"Duplicate value in sample {sample.keys()} after rename.")
                    continue
                new_sample[new_name] = value
            if self.must_match and not all(matched.values()):
                raise ValueError(f"Not all patterns ({matched}) matched sample keys ({sample.keys()}).")
 
            yield new_sample

    def __len__(self) -> int:
        return len(self.source_datapipe)


@functional_datapipe("extract_keys")
class ExtractKeysIterDataPipe(IterDataPipe[Dict]):
    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]], *args, duplicate_is_error=True, ignore_missing=False) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe
        self.duplicate_is_error = duplicate_is_error
        self.patterns = args
        self.ignore_missing = ignore_missing

    def __iter__(self) -> Iterator[Dict]:
        for sample in self.source_datapipe:
            result = []
            for pattern in self.patterns:
                matches = [x for x in sample.keys() if fnmatch(x, pattern)]
                if len(matches) == 0:
                    if self.ignore_missing:
                        continue
                    else:
                        raise ValueError(f"Cannot find {pattern} in sample keys {sample.keys()}.")
                if len(matches) > 1 and self.duplicate_is_error:
                    raise ValueError(f"Multiple sample keys {sample.keys()} match {pattern}.")
                value = sample[matches[0]]
                result.append(value)
            yield tuple(result)

    def __len__(self) -> int:
        return len(self.source_datapipe)
