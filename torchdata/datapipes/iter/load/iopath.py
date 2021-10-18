# Copyright (c) Facebook, Inc. and its affiliates.
import os

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


class IoPathFileListerIterDataPipe(IterDataPipe):
    r""":class:`IoPathFileListerIterDataPipe`.

    Iterable DataPipe to list the contents of the directory at the provided
    `root` URI.
    pathnames. This yields the full URI for each file within the directory.
    Args:
        root: The base URI directory to list files from
    """

    def __init__(self, *, root) -> None:
        try:
            from iopath.common.file_io import g_pathmgr
        except ImportError:
            raise ModuleNotFoundError(
                "Package `iopath` is required to be installed to use this "
                "datapipe. Please use `pip install iopath` or `conda install "
                "iopath`"
                "to install the package"
            )

        self.root = root
        self.pathmgr = g_pathmgr

    def __iter__(self):
        if self.pathmgr.isfile(self.root):
            yield self.root
        else:
            for file_name in self.pathmgr.ls(self.root):
                yield os.path.join(self.root, file_name)


@functional_datapipe("load_file_by_iopath")
class IoPathFileLoaderIterDataPipe(IterDataPipe):
    r""":class:`IoPathFileLoaderIterDataPipe`.

    Iterable DataPipe to load files from input datapipe which contains
    URIs. This yields a tuple of pathname and an opened filestream.
    Args:
        source_datapipe: Iterable DataPipe that provides the pathname
        mode: Specifies the mode in which the file is opened. This arg will be
            passed into `iopath.common.file_io.g_pathmgr.open` (internal only).
            Check each subclass of `PathHandler` to determine which modes are
            supported.
    """

    def __init__(self, source_datapipe, mode="r") -> None:
        try:
            from iopath.common.file_io import g_pathmgr
        except ImportError:
            raise ModuleNotFoundError(
                "Package `iopath` is required to be installed to use this "
                "datapipe. Please use `pip install iopath` or `conda install "
                "iopath`"
                "to install the package"
            )

        self.source_datapipe = source_datapipe
        self.pathmgr = g_pathmgr
        self.mode = mode

    def __iter__(self):
        for file_uri in self.source_datapipe:
            with self.pathmgr.open(file_uri, self.mode) as file:
                yield file_uri, file

    def __len__(self):
        return len(self.source_datapipe)
