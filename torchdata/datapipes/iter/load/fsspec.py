# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import posixpath

from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

from torch.utils.data.datapipes.utils.common import match_masks

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

try:
    import fsspec

except ImportError:
    fsspec = None

U = Union[bytes, bytearray, str]


def _assert_fsspec() -> None:
    if fsspec is None:
        raise ModuleNotFoundError(
            "Package `fsspec` is required to be installed to use this datapipe."
            "Please use `pip install fsspec` or `conda install -c conda-forge fsspec`"
            "to install the package"
        )


class FSSpecFileListerIterDataPipe(IterDataPipe[str]):
    r"""
    Lists the contents of the directory at the provided ``root`` pathname or URL,
    and yields the full pathname or URL for each file within the directory.

    Args:
        root: The root `fsspec` path directory or list of path directories to list files from
        masks: Unix style filter string or string list for filtering file name(s)

    Example:
        >>> from torchdata.datapipes.iter import FSSpecFileLister
        >>> datapipe = FSSpecFileLister(root=dir_path)
    """

    def __init__(
        self,
        root: Union[str, Sequence[str], IterDataPipe],
        masks: Union[str, List[str]] = "",
    ) -> None:
        _assert_fsspec()

        if isinstance(root, str):
            root = [
                root,
            ]
        if not isinstance(root, IterDataPipe):
            self.datapipe: IterDataPipe = IterableWrapper(root)  # type: ignore[assignment]
        else:
            self.datapipe = root
        self.masks = masks

    def __iter__(self) -> Iterator[str]:
        for root in self.datapipe:
            fs, path = fsspec.core.url_to_fs(root)

            if isinstance(fs.protocol, str):
                protocol_list = [fs.protocol]
            else:
                protocol_list = fs.protocol

            is_local = fs.protocol == "file" or not any(root.startswith(protocol) for protocol in protocol_list)
            if fs.isfile(path):
                yield root
            else:
                for file_name in fs.ls(path):
                    if not match_masks(file_name, self.masks):
                        continue

                    # ensure the file name has the full fsspec protocol path
                    if any(file_name.startswith(protocol) for protocol in protocol_list):
                        yield file_name
                    else:
                        if is_local:
                            abs_path = os.path.join(path, file_name)
                        else:
                            abs_path = posixpath.join(path, file_name)

                        starts_with = False
                        for protocol in protocol_list:
                            if root.startswith(protocol):
                                starts_with = True
                                yield protocol + "://" + abs_path
                                break

                        if not starts_with:
                            yield abs_path

    def list_files(self) -> List[str]:
        return [fp for fp in self]


@functional_datapipe("open_files_by_fsspec")
class FSSpecFileOpenerIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Opens files from input datapipe which contains `fsspec` paths and yields a tuple of
    pathname and opened file stream (functional name: ``open_files_by_fsspec``).

    Args:
        source_datapipe: Iterable DataPipe that provides the pathnames or URLs
        mode: An optional string that specifies the mode in which the file is opened (``"r"`` by default)

    Example:
        >>> from torchdata.datapipes.iter import FSSpecFileLister
        >>> datapipe = FSSpecFileLister(root=dir_path)
        >>> file_dp = datapipe.open_files_by_fsspec()
    """

    def __init__(self, source_datapipe: IterDataPipe[str], mode: str = "r") -> None:
        _assert_fsspec()

        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.mode: str = mode

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for file_uri in self.source_datapipe:
            fs, path = fsspec.core.url_to_fs(file_uri)
            file = fs.open(path, self.mode)
            yield file_uri, StreamWrapper(file)

    def __len__(self) -> int:
        return len(self.source_datapipe)


# Register for functional API for backward compatibility
IterDataPipe.register_datapipe_as_function("open_file_by_fsspec", FSSpecFileOpenerIterDataPipe)


@functional_datapipe("save_by_fsspec")
class FSSpecSaverIterDataPipe(IterDataPipe[str]):
    r"""
    Takes in a DataPipe of tuples of metadata and data, saves the data to the target
    path (generated by the filepath_fn and metadata), and yields the resulting `fsspec`
    path (functional name: ``save_by_fsspec``).

    Args:
        source_datapipe: Iterable DataPipe with tuples of metadata and data
        mode: Mode in which the file will be opened for write the data (``"w"`` by default)
        filepath_fn: Function that takes in metadata and returns the target path of the new file

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def filepath_fn(name: str) -> str:
        >>>     return dir_path + name
        >>> name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2", "3.txt": b"DATA3"}
        >>> source_dp = IterableWrapper(sorted(name_to_data.items()))
        >>> fsspec_saver_dp = source_dp.save_by_fsspec(filepath_fn=filepath_fn, mode="wb")
        >>> res_file_paths = list(fsspec_saver_dp)
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[Any, U]],
        mode: str = "w",
        filepath_fn: Optional[Callable] = None,
    ):
        _assert_fsspec()

        self.source_datapipe: IterDataPipe[Tuple[Any, U]] = source_datapipe
        self.mode: str = mode
        self.filepath_fn: Optional[Callable] = filepath_fn

    def __iter__(self) -> Iterator[str]:
        for meta, data in self.source_datapipe:
            filepath = meta if self.filepath_fn is None else self.filepath_fn(meta)
            fs, path = fsspec.core.url_to_fs(filepath)
            with fs.open(path, self.mode) as f:
                f.write(data)
            yield filepath

    def __len__(self) -> int:
        return len(self.source_datapipe)
