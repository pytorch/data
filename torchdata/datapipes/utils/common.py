# Copyright (c) Facebook, Inc. and its affiliates.
import os
import fnmatch
import tempfile
import warnings

from io import IOBase
from typing import Iterable, List, Tuple, Union


def match_masks(name: str, masks: Union[str, List[str]]) -> bool:
    # empty mask matches any input name
    if not masks:
        return True

    if isinstance(masks, str):
        return fnmatch.fnmatch(name, masks)

    for mask in masks:
        if fnmatch.fnmatch(name, mask):
            return True
    return False


def get_file_pathnames_from_root(
    root: str, masks: Union[str, List[str]], recursive: bool = False, abspath: bool = False
) -> Iterable[str]:

    # print out an error message and raise the error out
    def onerror(err: OSError):
        warnings.warn(err.filename + " : " + err.strerror)
        raise err

    for path, _, files in os.walk(root, onerror=onerror):
        if abspath:
            path = os.path.abspath(path)
        for f in files:
            if match_masks(f, masks):
                yield os.path.join(path, f)
        if not recursive:
            break


def get_file_binaries_from_pathnames(pathnames: Iterable, mode: str):
    if not isinstance(pathnames, Iterable):
        pathnames = [
            pathnames,
        ]

    if mode in ("b", "t"):
        mode = "r" + mode

    for pathname in pathnames:
        if not isinstance(pathname, str):
            raise TypeError("Expected string type for pathname, but got {}".format(type(pathname)))
        yield (pathname, open(pathname, mode))


def _default_filepath_fn(data):
    # Cross-platform Temporary Directory
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, os.path.basename(data))


def _default_cache_check_fn(data):
    return os.path.exists(data[0])


def validate_pathname_binary_tuple(data: Tuple[str, IOBase]):
    if not isinstance(data, tuple):
        raise TypeError(f"pathname binary data should be tuple type, but it is type {type(data)}")
    if len(data) != 2:
        raise TypeError(f"pathname binary stream tuple length should be 2, but got {len(data)}")
    if not isinstance(data[0], str):
        raise TypeError(f"pathname within the tuple should have string type pathname, but it is type {type(data[0])}")
    if not isinstance(data[1], IOBase):
        raise TypeError(
            f"binary stream within the tuple should have IOBase or"
            f"its subclasses as type, but it is type {type(data[1])}"
        )
