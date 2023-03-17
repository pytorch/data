# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys


def collect_init_dps(init_file_location):
    init_dps = set()
    with open(init_file_location) as init_file:
        while (line := init_file.readline()) != "":
            if line.startswith("__all__ "):
                while (line := init_file.readline()) != "" and (stripped_line := line.strip()).startswith('"'):
                    init_dps.add(stripped_line.replace(",", "").replace('"', ""))
                break
    return init_dps


def collect_rst_dps(rst_file_location):
    rst_dps = set()
    with open(rst_file_location) as rst_file:
        while (line := rst_file.readline()) != "":
            if line.count("class_template.rst") > 0 or line.count("function.rst") > 0:
                rst_file.readline()
                while (line := rst_file.readline()) != "" and len(stripped_line := line.strip()) > 1:
                    rst_dps.add(stripped_line)
    return rst_dps


def compare_sets(set_a, set_b, ignore_set=None):
    res = set_a.difference(set_b)
    if ignore_set is not None:
        res.difference_update(ignore_set)
    return res


def main():
    datapipes_folder = os.path.join("torchdata", "datapipes")
    init_file = "__init__.py"
    docs_source_folder = os.path.join("docs", "source")
    exit_code = 0

    for target, ignore_set in zip(["iter", "map", "utils"], [{"IterDataPipe", "Extractor"}, {"MapDataPipe"}, {}]):
        init_path = os.path.join(datapipes_folder, target, init_file)
        rst_path = os.path.join(docs_source_folder, "torchdata.datapipes." + target + ".rst")

        init_set = collect_init_dps(init_path)
        rst_set = collect_rst_dps(rst_path)

        dif_init = compare_sets(init_set, rst_set, ignore_set)
        dif_rst = compare_sets(rst_set, init_set)

        for elem in dif_init:
            print(f"Please add {elem} to {rst_path}")
            exit_code = 1
        for elem in dif_rst:
            print(f"{elem} is present in {rst_path} but not in {init_path}")
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
