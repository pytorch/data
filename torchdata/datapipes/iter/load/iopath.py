# Copyright (c) Facebook, Inc. and its affiliates.
import os

from torch.utils.data import IterDataPipe, functional_datapipe


class IoPathFileListerIterDataPipe(IterDataPipe):
    def __init__(self, *, root):
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
    def __init__(self, source_datapipe):
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

    def __iter__(self):
        for file_name in self.source_datapipe:
            with self.pathmgr.open(file_name) as file:
                yield file

    def __len__(self):
        return len(self.source_datapipe)
