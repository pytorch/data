# Copyright (c) Facebook, Inc. and its affiliates.
import hashlib

from io import IOBase
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from typing import Dict, Iterator, Tuple, Union


D_type = Union[str, bytes, bytearray]
U = Union[D_type, StreamWrapper]


@functional_datapipe("check_hash")
class HashCheckerIterDataPipe(IterDataPipe[Tuple[str, U]]):
    r"""
    Iterable DataPipe that computes and checks the hash of each file, from an input
    DataPipe of tuples of file name and data (stream). If the hashes match the given hash
    in the dictionary, it yields a tuple of file name and data (stream). Otherwise, it raises an error.

    Args:
        source_datapipe: IterDataPipe with tuples of file name and data (stream)
        hash_dict: Dict that maps file names to their corresponding hashes
        hash_type: The type of hash function to apply
        rewind: Rewind the stream after using the stream to compute the hash (this
            does not work with non-seekable stream, e.g. HTTP)

    Usage: dp = dp.check_hash({'train.py':'0d8b94d9fa9fb1ad89b9e3da9e1521495dca558fc5213b0fd7fd7b71c23f9921'})
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[str, IOBase]],
        hash_dict: Dict[str, str],
        hash_type: str = "sha256",
        rewind: bool = True,
    ) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IOBase]] = source_datapipe
        self.hash_dict: Dict[str, str] = hash_dict
        self.hash_type: str = hash_type
        self.rewind: bool = rewind

        if self.hash_type not in ["sha256", "md5"]:
            raise ValueError("Invalid hash_type requested, should be one of {}".format(["sha256", "md5"]))

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for file_name, data in self.source_datapipe:
            if self.hash_type == "sha256":
                hash_func = hashlib.sha256()
            else:
                hash_func = hashlib.md5()

            if isinstance(data, (str, bytes, bytearray)):
                if isinstance(data, str):
                    data = data.decode()
                hash_func.update(data)
            # File Stream
            else:
                # Not all of streams have `read(bytes)` method.
                # `__iter__` method is chosen because it is a common interface for IOBase.
                for d in data:
                    hash_func.update(d)

                # TODO(133): this will not work (or work crappy for non-seekable steams like http)
                if self.rewind:
                    data.seek(0)

            if file_name not in self.hash_dict:
                raise RuntimeError("Unspecified hash for file {}".format(file_name))

            if hash_func.hexdigest() != self.hash_dict[file_name]:
                raise RuntimeError(
                    "The hash {} of {} does not match. Delete the file manually and retry.".format(
                        hash_func.hexdigest(), file_name
                    )
                )

            if isinstance(data, (str, bytes, bytearray)):
                yield file_name, data
            else:
                yield file_name, StreamWrapper(data)

    def __len__(self) -> int:
        return len(self.source_datapipe)
