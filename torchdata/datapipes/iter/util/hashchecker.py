# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib

from io import IOBase
from typing import Dict, Iterator, Tuple, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper


D_type = Union[str, bytes, bytearray]
U = Union[D_type, StreamWrapper]


@functional_datapipe("check_hash")
class HashCheckerIterDataPipe(IterDataPipe[Tuple[str, U]]):
    r"""
    Computes and checks the hash of each file, from an input DataPipe of tuples of file name and
    data/stream (functional name: ``check_hash``). If the hashes match the given hash
    in the dictionary, it yields a tuple of file name and data/stream. Otherwise, it will raise an error.

    Args:
        source_datapipe: IterDataPipe with tuples of file name and data/stream
        hash_dict: Dictionary that maps file names to their corresponding hashes
        hash_type: The type of hash function to apply
        rewind: Rewind the stream after using the stream to compute the hash (this
            does not work with non-seekable stream, e.g. HTTP)

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, FileOpener
        >>> expected_MD5_hash = "bb9675028dd39d2dd2bf71002b93e66c"
        File is from "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        >>> file_dp = FileOpener(IterableWrapper(["LICENSE.txt"]), mode='rb')
        >>> # An exception is only raised when the hash doesn't match, otherwise (path, stream) is returned
        >>> check_hash_dp = file_dp.check_hash({"LICENSE.txt": expected_MD5_hash}, "md5", rewind=True)
        >>> reader_dp = check_hash_dp.readlines()
        >>> it = iter(reader_dp)
        >>> path, line = next(it)
        >>> path
        LICENSE.txt
        >>> line
        b'BSD 3-Clause License'
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
                # Not all streams have `read(bytes)` method.
                # `__iter__` method is chosen because it is a common interface for IOBase.
                for d in data:
                    hash_func.update(d)

                # TODO(133): this will not work (or work crappy for non-seekable steams like http)
                if self.rewind:
                    data.seek(0)

            if file_name not in self.hash_dict:
                raise RuntimeError(f"Unspecified hash for file {file_name}")

            if hash_func.hexdigest() != self.hash_dict[file_name]:
                raise RuntimeError(
                    f"The computed hash {hash_func.hexdigest()} of {file_name} does not match the expected"
                    f"hash {self.hash_dict[file_name]}. Delete the file manually and retry."
                )

            if isinstance(data, (str, bytes, bytearray)):
                yield file_name, data
            else:
                yield file_name, StreamWrapper(data)

    def __len__(self) -> int:
        return len(self.source_datapipe)
