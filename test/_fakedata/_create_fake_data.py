# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import tarfile


NUMBER_OF_FILES = 3
FILES = [
    ("bytes", "bt", "{fn}_0123456789abcdef\n", True),
    ("csv", "csv", "key,item\n0,{fn}_0\n1,{fn}_1\n"),
    ("json", "json", '{{"{fn}_0": [{{"{fn}_01": 1}}, {{"{fn}_02": 2}}], "{fn}_1": 1}}\n'),
    ("txt", "txt", "{fn}_0123456789abcdef\n"),
]


def create_files(folder, suffix, data, encoding=False):
    os.makedirs(folder, exist_ok=True)
    for i in range(NUMBER_OF_FILES):
        fn = str(i)
        d = data.format(fn=fn)
        mode = "wb" if encoding else "wt"
        if encoding:
            d = d.encode()
        with open(folder + "/" + fn + "." + suffix, mode) as f:
            f.write(d)

    with tarfile.open(folder + ".tar", mode="w") as archive:
        archive.add(folder)

    with tarfile.open(folder + ".tar.gz", mode="w:gz") as archive:
        archive.add(folder)


def add_data_to_tar(archive, name, value):
    if isinstance(value, str):
        value = value.encode()
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def create_wds_tar(dest):
    with tarfile.open(dest, mode="w") as archive:
        for i in range(10):
            add_data_to_tar(archive, f"data/{i}.txt", f"text{i}")
            add_data_to_tar(archive, f"data/{i}.bin", f"bin{i}")


if __name__ == "__main__":
    create_wds_tar("wds.tar")
    for args in FILES:
        create_files(*args)
