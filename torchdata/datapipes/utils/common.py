# Copyright (c) Facebook, Inc. and its affiliates.
import os
import tempfile

from io import IOBase
from torch.utils.data.datapipes.utils.common import StreamWrapper
from typing import Tuple


def _default_filepath_fn(data):
    # Cross-platform Temporary Directory
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, os.path.basename(data))
    return os.path.normpath(data)


def validate_pathname_binary_tuple(data: Tuple[str, IOBase]):
    if not isinstance(data, tuple):
        raise TypeError(f"pathname binary data should be tuple type, but it is type {type(data)}")
    if len(data) != 2:
        raise TypeError(f"pathname binary stream tuple length should be 2, but got {len(data)}")
    if not isinstance(data[0], str):
        raise TypeError(f"pathname within the tuple should have string type pathname, but it is type {type(data[0])}")
    if not isinstance(data[1], IOBase) and not isinstance(data[1], StreamWrapper):
        raise TypeError(
            f"binary stream within the tuple should have IOBase or"
            f"its subclasses as type, but it is type {type(data[1])}"
        )
