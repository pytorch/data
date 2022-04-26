import re
from typing import Any, Dict, Iterator, List, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


def pathsplit(p):
    """Split a path into the basename and the extensions."""
    if "." not in p:
        return p, ""
    # convert Windows pathnames to UNIX pathnames, otherwise
    # we get an inconsistent mix of the Windows path to the tar
    # file followed by the POSIX path inside that tar file
    p = re.sub(r"[\\]", "/", p)
    # we need to use a regular expression because os.path is
    # platform specific, but tar files always contain POSIX paths
    prefix, suffix = re.search(r"^(.*?)(\.[^/]*)$", p).groups()
    return prefix, suffix


@functional_datapipe("webdataset")
class WebDatasetIterDataPipe(IterDataPipe[Dict]):
    r"""
    Iterable DataPipe that accepts stream of (path, data) tuples, usually,
    representing the pathnames and files of a tar archive (functional name:
    ``webdataset''). This aggregates consecutive items with the same basename
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
