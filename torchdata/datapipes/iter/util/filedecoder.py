# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import re
from fnmatch import fnmatch
from typing import Any, Dict, Iterator, Tuple

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


def _decode_bin(stream):
    return stream.read()


def _decode_text(stream):
    binary = stream.read()
    return binary.decode("utf-8")


def _decode_pickle(stream):
    return pickle.load(stream)


default_decoders = [
    ("*.bin", _decode_bin),
    ("*.txt", _decode_text),
    ("*.pyd", _decode_pickle),
]


def _find_decoder(decoders, path):
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
            decoder = _find_decoder(self.decoders, path)
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
