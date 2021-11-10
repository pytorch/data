# Copyright (c) Facebook, Inc. and its affiliates.
import os

from typing import Any, Callable, Iterator, List, Tuple, Union

from torch.utils.data.datapipes.utils.common import match_masks

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import _default_filepath_fn

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
    r""":class:`IoPathFileListerIterDataPipe`.

    Iterable DataPipe to list the contents of the directory at the provided `root` pathname or url,
    and yields the full pathname or url for each file within the directory.

    Args:
        root: The root local filepath or url directory to list files from
        masks: Unix style filter string or string list for filtering file name(s)
        pathmgr: Custom iopath PathManager. If not specified, a default PathManager is created.

    Note:
        Default PathManager currently supports local file path, normal HTTP url and OneDrive url.
        S3 url is supported only with `iopath`>=0.1.9.
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


@functional_datapipe("load_file_by_iopath")
class IoPathFileLoaderIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r""":class:`IoPathFileLoaderIterDataPipe`.

    Iterable DataPipe to open files from input datapipe which contains pathnames or URLs,
    and yields a tuple of pathname and opened file stream.

    Args:
        source_datapipe: Iterable DataPipe that provides the pathnames or urls
        mode: An optional string that specifies the mode in which the file is opened ('r' by default)
        pathmgr: Custom iopath PathManager. If not specified, a default PathManager is created.

    Note:
        Default PathManager currently supports local file path, normal HTTP url and OneDrive url.
        S3 url is supported only with `iopath`>=0.1.9.
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
            with self.pathmgr.open(file_uri, self.mode) as file:
                yield file_uri, StreamWrapper(file)

    def __len__(self) -> int:
        return len(self.source_datapipe)


@functional_datapipe("save_by_iopath")
class IoPathSaverIterDataPipe(IterDataPipe[str]):
    r"""
    Iterable DataPipe that takes in a DataPipe of tuples of metadata and data, saves the data
    to the target path (generated by the filepath_fn and metadata), and yields the resulting path
    by iopath

    Args:
        source_datapipe: Iterable DataPipe with tuples of metadata and data
        mode: Mode in which the file will be opened for write the data ("w" by default)
        filepath_fn: Function that takes in metadata nad returns the target path of the new file
        pathmgr: Custom iopath PathManager. If not specified, a default PathManager is created.

    Note:
        Default PathManager currently supports local file path, normal HTTP url and OneDrive url.
        S3 url is supported only with `iopath`>=0.1.9.
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[Any, U]],
        mode: str = "w",
        filepath_fn: Callable[[Any], str] = _default_filepath_fn,
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
        self.fn: Callable[[Any], str] = filepath_fn
        self.pathmgr = _create_default_pathmanager() if pathmgr is None else pathmgr

    def __iter__(self) -> Iterator[str]:
        for meta, data in self.source_datapipe:
            filepath = self.fn(meta)
            with self.pathmgr.open(filepath, self.mode) as f:
                f.write(data)
            yield filepath

    def register_handler(self, handler, allow_override=False):
        self.pathmgr.register_handler(handler, allow_override=allow_override)

    def __len__(self) -> int:
        return len(self.source_datapipe)
