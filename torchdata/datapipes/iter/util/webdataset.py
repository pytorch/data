# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import urllib.parse
from fnmatch import fnmatch
from typing import Any, Dict, Iterator, List, Tuple, Union

from torch.utils.data.datapipes.utils.common import StreamWrapper

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


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
    prefix = s[: m.start()]
    rest = shardexpand(s[m.end() :])
    rng = s[m.start() + 1 : m.end() - 1]
    lohi = rng.split("..")
    if len(lohi[0]) != len(lohi[1]):
        raise ValueError("Shard specifications must have " + f"same number of digits for low and high values in {s}.")
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
    r"""
    Expands incoming shard strings into shards.

    Sharded data files are named using shell-like brace notation. For example,
    an ImageNet dataset sharded into 1200 shards and stored on a web server
    might be named `imagenet-{000000..001199}.tar`.

    Note that shard names can be expanded without any server transactions;
    this makes `shardexpand` reproducible and storage system independent
    (unlike `ListFiles` etc.).

    Args:
        source_datapipe: a DataPipe yielding a stream of  pairs

    Returns:
        a DataPipe yielding a stream of expanded pathnames.
    """

    def __init__(self, source_datapipe: IterDataPipe[str]) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[str] = source_datapipe

    def __iter__(self) -> Iterator[str]:
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
    r"""
    Decode files in `(fname, stream)` tuples based on filename extensions.

    Args:
        source_datapipe: a DataPipe yielding a stream of pairs, as returned by `load_from_tar`
        *args: pairs of the form `("*.jpg", imread)`
        **kw: arguments of the form `jpg=imread`, shorthand for `("*.jpg", imread)`
        must_decode: require an decoder for every file encountered (True)
        defaults: list of default decoders (prepended to `args`)

    Returns:
        a DataPipe yielding a stream of `(fname, data)` pairs

    Examples:
        >>> from torchdata.datapipes.iter import FileLister, FileOpener
        >>> from imageio import imread
        >>>
        >>> dp = FileLister("data", "imagenet-*.tar").open().load_from_tar().decode(jpg=imread)
        >>> for path, data in dataset:
        >>>     if path.endswith(".jpg"):
        >>>         imshow(data)
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[str, Any]],
        *args,
        must_decode=True,
        defaults=default_decoders,
        **kw,
    ) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[Tuple[str, Any]] = source_datapipe
        self.must_decode = must_decode
        self.decoders = list(defaults) + list(args)
        self.decoders += [("*." + k, v) for k, v in kw.items()]

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
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


@functional_datapipe("rename_keys")
class RenameKeysIterDataPipe(IterDataPipe[Dict]):
    r"""
    Given a stream of dictionaries, rename keys using glob patterns.

    Args:
        source_datapipe: a DataPipe yielding a stream of dictionaries.
        keep_unselected: keep keys/value pairs even if they don't match any pattern (False)
        must_match: all key value pairs must match (True)
        duplicate_is_error: it is an error if two renamings yield the same key (True)
        *args: `(renamed, pattern)` pairs
        **kw: `renamed=pattern` pairs

    Returns:
        a DataPipe yielding a stream of dictionaries.

    Examples:
        >>> dp = IterableWrapper([{"/a/b.jpg": b"data"}]).rename_keys(image="*.jpg")
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[List[Union[Dict, List]]],
        *args,
        keep_unselected=False,
        must_match=True,
        duplicate_is_error=True,
        **kw,
    ) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe
        self.must_match = must_match
        self.keep_unselected = keep_unselected
        self.duplicate_is_error = duplicate_is_error
        self.renamings = [(pattern, output) for output, pattern in args]
        self.renamings += [("*." + pattern, output) for output, pattern in kw.items()]

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
    r"""
    Given a stream of dictionaries, return a stream of tuples by selecting keys using glob patterns.

    Args:
        source_datapipe: a DataPipe yielding a stream of dictionaries.
        duplicate_is_error: it is an error if the same key is selected twice (True)
        ignore_missing: skip any dictionaries where one or more patterns don't match (False)
        *args: list of glob patterns or list of glob patterns

    Returns:
        a DataPipe yielding a stream of tuples

    Examples:
        >>> dp = FileLister(...).load_from_tar().webdataset().decode(...).extract_keys(["*.jpg", "*.png"], "*.gt.txt")
    """

    def __init__(
        self, source_datapipe: IterDataPipe[Dict], *args, duplicate_is_error=True, ignore_missing=False
    ) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[Dict] = source_datapipe
        self.duplicate_is_error = duplicate_is_error
        self.patterns = args
        self.ignore_missing = ignore_missing

    def __iter__(self) -> Iterator[Tuple]:
        for sample in self.source_datapipe:
            result = []
            for pattern in self.patterns:
                pattern = [pattern] if not isinstance(pattern, (list, tuple)) else pattern
                matches = [x for x in sample.keys() if any(fnmatch(x, p) for p in pattern)]
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


def _pick(buf, rng):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample


@functional_datapipe("incshuffle")
class IncrementalShufflerIterDataPipe(IterDataPipe[Dict]):
    r"""
    Perform incremental shuffling on a stream of data.

    This initially reads `initial` samples. Subsequently, an output sample
    is generated by randomly selecting an input sample from the buffer and
    replacing it with another sample from the input stream. If the shuffle
    buffer is smaller than `buffer_size`, an additional sample is used to fill
    up the shuffle buffer.

    This shuffle function allows the user to make a tradeoff between startup
    latency and randomness.

    Args:
        source_datapipe: a DataPipe yielding a stream of samples
        rng: user supplied random number generator
        initial: initial buffer size (10)
        buffer_size: buffer size for shuffling (1000)

    Returns:
        a DataPipe yielding a stream of tuples
    """

    def __init__(self, source_datapipe: IterDataPipe[Any], rng=None, initial=10, buffer_size=1000):
        super().__init__()
        self.source_datapipe: IterDataPipe[Any] = source_datapipe
        self.rng = rng or random.Random(os.urandom(8))
        self.initial = initial
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Any]:
        initial = min(self.initial, self.buffer_size)
        buf = []
        data = iter(self.source_datapipe)
        for sample in data:
            buf.append(sample)
            if len(buf) < self.buffer_size:
                try:
                    buf.append(next(data))  # skipcq: PYL-R1708
                except StopIteration:
                    pass
            if len(buf) >= initial:
                yield _pick(buf, self.rng)
        while len(buf) > 0:
            yield _pick(buf, self.rng)

    def __len__(self) -> int:
        return len(self.source_datapipe)


def cache_by_fname(s):
    result = re.sub("^.*/", "", s)
    return urllib.parse.quote(result)


if os.name == "nt":
    default_cachedir = "datacache"
else:
    default_cachedir = "_datacache"


@functional_datapipe("filecache")
class FileCacheIterDataPipe(IterDataPipe[Dict]):
    r""" """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[str, Any]],
        cachedir=default_cachedir,
        cachename=cache_by_fname,
        chunksize=1024**2,
        verbose=False,
        makedir=True,
    ) -> None:
        super().__init__()
        if not os.path.exists(cachedir):
            if makedir:
                os.makedirs(cachedir)
            else:
                raise ValueError(f"Cache directory {cachedir} does not exist.")
        self.source_datapipe: IterDataPipe[Tuple[str, Any]] = source_datapipe
        self.cachedir = cachedir
        self.cachename = cachename
        self.verbose = verbose
        self.chunksize = chunksize

    def __iter__(self) -> Iterator[Dict]:
        for url, stream in self.source_datapipe:
            cached = os.path.join(self.cachedir, self.cachename(url))
            if not os.path.exists(cached):
                if self.verbose:
                    print(f"# downloading {url} to {cached}", file=sys.stderr)
                with open(cached + ".temp", "wb") as dest:
                    shutil.copyfileobj(stream, dest, self.chunksize)
                os.rename(cached + ".temp", cached)
            if self.verbose:
                print(f"# returning {cached}", file=sys.stderr)
            cached_stream = open(cached, "rb")
            yield url, StreamWrapper(cached_stream)

    def __len__(self) -> int:
        return len(self.source_datapipe)
