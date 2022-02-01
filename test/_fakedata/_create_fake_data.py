# Copyright (c) Facebook, Inc. and its affiliates.
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


if __name__ == "__main__":
    for args in FILES:
        create_files(*args)
