# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

from torch.utils.data.datapipes.utils.common import match_masks

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

try:
    import iopath

except ImportError:
    iopath = None

U = Union[bytes, bytearray, str]


def _create_default_pathmanager():
    from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathManager

    pathmgr = PathManager()
    pathmgr.register_handler(HTTPURLHandler(), allow_override=True)
    pathmgr.register_handler(OneDrivePathHandler(), allow_override=True)
    # S3PathHandler is not included in 0.1.8
    try:
        from iopath.common.s3 import S3PathHandler

        pathmgr.register_handler(S3PathHandler(), allow_override=True)
    except ImportError:
        pass
    return pathmgr


class IoPathFileListerIterDataPipe(IterDataPipe[str]):
    r"""
    Lists the contents of the directory at the provided ``root`` pathname or URL,
    and yields the full pathname or URL for each file within the directory.

    Args:
        root: The root local filepath or URL directory to list files from
        masks: Unix style filter string or string list for filtering file name(s)
        pathmgr: Custom ``iopath.PathManager``. If not specified, a default ``PathManager`` is created.

    Note:
        Default ``PathManager`` currently supports local file path, normal HTTP URL and OneDrive URL.
        S3 URL is supported only with ``iopath``>=0.1.9.

    Example:
        >>> from torchdata.datapipes.iter import IoPathFileLister
        >>> datapipe = IoPathFileLister(root=S3URL)
    """

    def __init__(
        self,
        root: str,
        masks: Union[str, List[str]] = "",
        *,
        pathmgr=None,
    ) -> None:
        if iopath is None:
            raise ModuleNotFoundError(
                "Package `iopath` is required to be installed to use this datapipe."
                "Please use `pip install iopath` or `conda install -c conda-forge iopath`"
                "to install the package"
            )

        self.root: str = root
        self.pathmgr = _create_default_pathmanager() if pathmgr is None else pathmgr
        self.masks = masks

    def register_handler(self, handler, allow_override=False):
        self.pathmgr.register_handler(handler, allow_override=allow_override)

    def __iter__(self) -> Iterator[str]:
        if self.pathmgr.isfile(self.root):
            yield self.root
        else:
            for file_name in self.pathmgr.ls(self.root):
                if match_masks(file_name, self.masks):
                    yield os.path.join(self.root, file_name)


@functional_datapipe("open_file_by_iopath")
class IoPathFileOpenerIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Opens files from input datapipe which contains pathnames or URLs,
    and yields a tuple of pathname and opened file stream (functional name: ``open_file_by_iopath``).

    Args:
        source_datapipe: Iterable DataPipe that provides the pathnames or URLs
        mode: An optional string that specifies the mode in which the file is opened (``"r"`` by default)
        pathmgr: Custom ``iopath.PathManager``. If not specified, a default ``PathManager`` is created.

    Note:
        Default ``PathManager`` currently supports local file path, normal HTTP URL and OneDrive URL.
        S3 URL is supported only with `iopath`>=0.1.9.

    Example:
        >>> from torchdata.datapipes.iter import IoPathFileLister
        >>> datapipe = IoPathFileLister(root=S3URL)
        >>> file_dp = datapipe.open_file_by_iopath()
    """

    def __init__(self, source_datapipe: IterDataPipe[str], mode: str = "r", pathmgr=None) -> None:
        if iopath is None:
            raise ModuleNotFoundError(
                "Package `iopath` is required to be installed to use this datapipe."
                "Please use `pip install iopath` or `conda install -c conda-forge iopath`"
                "to install the package"
            )

        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.pathmgr = _create_default_pathmanager() if pathmgr is None else pathmgr
        self.mode: str = mode

    def register_handler(self, handler, allow_override=False):
        self.pathmgr.register_handler(handler, allow_override=allow_override)

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for file_uri in self.source_datapipe:
            file = self.pathmgr.open(file_uri, self.mode)
            yield file_uri, StreamWrapper(file)

    def __len__(self) -> int:
        return len(self.source_datapipe)


@functional_datapipe("save_by_iopath")
class IoPathSaverIterDataPipe(IterDataPipe[str]):
    r"""
    Takes in a DataPipe of tuples of metadata and data, saves the data
    to the target path which is generated by the ``filepath_fn`` and metadata, and yields the resulting path
    in `iopath` format (functional name: ``save_by_iopath``).

    Args:
        source_datapipe: Iterable DataPipe with tuples of metadata and data
        mode: Mode in which the file will be opened for write the data (``"w"`` by default)
        filepath_fn: Function that takes in metadata and returns the target path of the new file
        pathmgr: Custom ``iopath.PathManager``. If not specified, a default ``PathManager`` is created.

    Note:
        Default ``PathManager`` currently supports local file path, normal HTTP URL and OneDrive URL.
        S3 URL is supported only with `iopath`>=0.1.9.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def filepath_fn(name: str) -> str:
        >>>     return S3URL + name
        >>> name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2", "3.txt": b"DATA3"}
        >>> source_dp = IterableWrapper(sorted(name_to_data.items()))
        >>> iopath_saver_dp = source_dp.save_by_iopath(filepath_fn=filepath_fn, mode="wb")
        >>> res_file_paths = list(iopath_saver_dp)
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[Any, U]],
        mode: str = "w",
        filepath_fn: Optional[Callable] = None,
        *,
        pathmgr=None,
    ):
        if iopath is None:
            raise ModuleNotFoundError(
                "Package `iopath` is required to be installed to use this datapipe."
                "Please use `pip install iopath` or `conda install -c conda-forge iopath`"
                "to install the package"
            )

        self.source_datapipe: IterDataPipe[Tuple[Any, U]] = source_datapipe
        self.mode: str = mode
        self.filepath_fn: Optional[Callable] = filepath_fn
        self.pathmgr = _create_default_pathmanager() if pathmgr is None else pathmgr

    def __iter__(self) -> Iterator[str]:
        for meta, data in self.source_datapipe:
            filepath = meta if self.filepath_fn is None else self.filepath_fn(meta)
            with iopath.file_lock(filepath):
                if not os.path.exists(filepath):
                    with self.pathmgr.open(filepath, self.mode) as f:
                        f.write(data)
            yield filepath

    def register_handler(self, handler, allow_override=False):
        self.pathmgr.register_handler(handler, allow_override=allow_override)

    def __len__(self) -> int:
        return len(self.source_datapipe)
