# Copyright (c) Facebook, Inc. and its affiliates.

import os
import tempfile

from torchdata.datapipes.iter import IterDataPipe
from typing import List, Tuple, TypeVar


T_co = TypeVar("T_co", covariant=True)


class IDP_NoLen(IterDataPipe):
    def __init__(self, input_dp) -> None:
        super().__init__()
        self.input_dp = input_dp

    def __iter__(self):
        for i in self.input_dp:
            yield i


def get_name(path_and_stream):
    return os.path.basename(path_and_stream[0]), path_and_stream[1]


# Given a DataPipe and integer n, iterate the DataPipe for n elements and store the elements into a list
# Then, reset the DataPipe and return a tuple of two lists
# 1. A list of elements yielded before the reset
# 2. A list of all elements of the DataPipe after the reset
def reset_after_n_next_calls(datapipe: IterDataPipe[T_co], n: int) -> Tuple[List[T_co], List[T_co]]:
    it = iter(datapipe)
    res_before_reset = []
    for _ in range(n):
        res_before_reset.append(next(it))
    return res_before_reset, list(datapipe)


def create_temp_dir(dir=None):
    # The temp dir and files within it will be released and deleted in tearDown().
    # Adding `noqa: P201` to avoid mypy's warning on not releasing the dir handle within this function.
    temp_dir = tempfile.TemporaryDirectory(dir=dir)  # noqa: P201
    return temp_dir


def create_temp_files(temp_dir, prefix=1, empty=True):
    temp_dir_path = temp_dir.name

    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, prefix=str(prefix), suffix=".txt") as f:
        temp_file1_name = f.name
    with open(temp_file1_name, "w") as f1:
        f1.write("0123456789abcdef")

    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, prefix=str(prefix + 1), suffix=".byte") as f:
        temp_file2_name = f.name
    with open(temp_file2_name, "wb") as f2:
        f2.write(b"0123456789abcdef")

    if empty:
        with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, prefix=str(prefix + 2), suffix=".empty") as f:
            temp_file3_name = f.name
        return temp_file1_name, temp_file2_name, temp_file3_name

    return temp_file1_name, temp_file2_name
