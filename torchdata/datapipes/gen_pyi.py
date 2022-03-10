# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch.utils.data.gen_pyi as core_gen_pyi
from torch.utils.data.gen_pyi import gen_from_template, get_method_definitions


def get_lines_base_file(base_file_path: str, to_skip: Optional[Set[str]] = None):
    with open(base_file_path) as f:
        lines = f.readlines()
        res = []
        if to_skip is None:
            return lines
        for line in lines:
            skip_flag = False
            for skip_line in to_skip:
                if skip_line in line:
                    skip_flag = True
            if not skip_flag:
                line = line.replace("\n", "")
                res.append(line)
        return res


def gen_pyi() -> None:
    ROOT_DIR = Path(__file__).parent.resolve()
    print(f"Generating DataPipe Python interface file in {ROOT_DIR}")

    iter_init_base = get_lines_base_file(
        os.path.join(ROOT_DIR, "iter/__init__.py"),
        {"from torch.utils.data import IterDataPipe", "# Copyright (c) Facebook, Inc. and its affiliates."},
    )

    # Core Definitions
    core_iter_method_definitions = get_method_definitions(
        core_gen_pyi.iterDP_file_path,
        core_gen_pyi.iterDP_files_to_exclude,
        core_gen_pyi.iterDP_deprecated_files,
        "IterDataPipe",
        core_gen_pyi.iterDP_method_to_special_output_type,
    )

    # TorchData Definitions
    iterDP_file_paths: List[str] = ["iter/load", "iter/transform", "iter/util"]
    iterDP_files_to_exclude: Set[str] = {"__init__.py"}
    iterDP_deprecated_files: Set[str] = set()
    iterDP_method_to_special_output_type: Dict[str, str] = {
        "bucketbatch": "IterDataPipe",
        "dataframe": "torcharrow.DataFrame",
        "end_caching": "IterDataPipe",
        "unzip": "List[IterDataPipe]",
        "read_from_tar": "IterDataPipe",
        "read_from_xz": "IterDataPipe",
        "read_from_zip": "IterDataPipe",
        "extract": "IterDataPipe",
    }
    method_name_exlusion: Set[str] = {"def extract", "read_from_tar", "read_from_xz", "read_from_zip"}

    td_iter_method_definitions = get_method_definitions(
        iterDP_file_paths,
        iterDP_files_to_exclude,
        iterDP_deprecated_files,
        "IterDataPipe",
        iterDP_method_to_special_output_type,
        root=str(pathlib.Path(__file__).parent.resolve()),
    )

    td_iter_method_definitions = [
        s for s in td_iter_method_definitions if all(ex not in s for ex in method_name_exlusion)
    ]

    iter_method_definitions = core_iter_method_definitions + td_iter_method_definitions

    replacements = [("${init_base}", iter_init_base, 0), ("${IterDataPipeMethods}", iter_method_definitions, 4)]

    gen_from_template(
        dir=str(ROOT_DIR),
        template_name="iter/__init__.pyi.in",
        output_name="iter/__init__.pyi",
        replacements=replacements,
    )
    # TODO: Add map_method_definitions when there are MapDataPipes defined in this library


if __name__ == "__main__":
    gen_pyi()
