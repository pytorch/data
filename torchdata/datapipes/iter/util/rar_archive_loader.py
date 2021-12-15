import io
import os.path
from typing import Tuple, Iterator
from unittest.mock import patch

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple


class RarfilePatcher:
    def __init__(self):
        from rarfile import DirectReader

        unpatched_read = DirectReader._read

        def patched_read(self, cnt):
            whence = self._cur.add_size - self._cur_avail
            self._fd.seek(self._inf.header_offset, whence)
            if whence == 0:
                self._cur = self._parser._parse_header(self._fd)
            return unpatched_read(self, cnt)

        self._patch = patch("rarfile.DirectReader._read", new=patched_read)

    def start(self):
        self._patch.start()

    def stop(self):
        self._patch.stop()


@functional_datapipe("load_from_rar")
class RarArchiveLoaderIterDataPipe(IterDataPipe[Tuple[str, io.BufferedIOBase]]):
    def __init__(self, datapipe: IterDataPipe[Tuple[str, io.BufferedIOBase]], *, length: int = -1):
        try:
            import rarfile
        except ImportError as error:
            raise ModuleNotFoundError(
                "Package `rarfile` is required to be installed to use this datapipe. "
                "Please use `pip install rarfile` or `conda -c conda-forge install rarfile` to install it."
            ) from error

        # check if at least one system library for reading rar archives is available to be used by rarfile
        rarfile.tool_setup()

        self.datapipe = datapipe
        self.length = length

    def __iter__(self) -> Iterator[Tuple[str, io.BufferedIOBase]]:
        import rarfile

        patcher = RarfilePatcher()
        patcher.start()

        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            path, stream = data

            rar = rarfile.RarFile(stream)
            for info in rar.infolist():
                if info.filename.endswith("/"):
                    continue

                inner_path = os.path.join(path, info.filename)
                file_obj = rar.open(info)

                yield inner_path, StreamWrapper(file_obj)  # type: ignore[misc]

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length
