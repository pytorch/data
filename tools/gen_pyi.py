# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch.utils.data.datapipes.gen_pyi as core_gen_pyi
from torch.utils.data.datapipes.gen_pyi import gen_from_template, get_method_definitions


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
    DATAPIPE_DIR = Path(__file__).parent.parent.resolve() / "torchdata" / "datapipes"
    print(f"Generating DataPipe Python interface file in {DATAPIPE_DIR}")

    # Base __init__ file
    iter_init_base = get_lines_base_file(
        os.path.join(DATAPIPE_DIR, "iter/__init__.py"),
        {"from torch.utils.data import IterDataPipe", "# Copyright (c) Facebook, Inc. and its affiliates."},
    )

    map_init_base = get_lines_base_file(
        os.path.join(DATAPIPE_DIR, "map/__init__.py"),
        {"from torch.utils.data import MapDataPipe", "# Copyright (c) Facebook, Inc. and its affiliates."},
    )

    # Core Definitions
    core_iter_method_definitions = get_method_definitions(
        core_gen_pyi.iterDP_file_path,
        core_gen_pyi.iterDP_files_to_exclude,
        core_gen_pyi.iterDP_deprecated_files,
        "IterDataPipe",
        core_gen_pyi.iterDP_method_to_special_output_type,
    )

    core_map_method_definitions = get_method_definitions(
        core_gen_pyi.mapDP_file_path,
        core_gen_pyi.mapDP_files_to_exclude,
        core_gen_pyi.mapDP_deprecated_files,
        "MapDataPipe",
        core_gen_pyi.mapDP_method_to_special_output_type,
    )

    # TorchData Definitions
    # IterDataPipes
    iterDP_file_paths: List[str] = ["iter/load", "iter/transform", "iter/util"]
    iterDP_files_to_exclude: Set[str] = {"__init__.py"}
    iterDP_deprecated_files: Set[str] = set()
    iterDP_method_to_special_output_type: Dict[str, str] = {
        "bucketbatch": "IterDataPipe",
        "dataframe": "torcharrow.DataFrame",
        "end_caching": "IterDataPipe",
        "unzip": "List[IterDataPipe]",
        "random_split": "Union[IterDataPipe, List[IterDataPipe]]",
        "read_from_tar": "IterDataPipe",
        "read_from_xz": "IterDataPipe",
        "read_from_zip": "IterDataPipe",
        "extract": "IterDataPipe",
        "to_map_datapipe": "MapDataPipe",
        "round_robin_demux": "List[IterDataPipe]",
    }
    iter_method_name_exclusion: Set[str] = {"def extract", "read_from_tar", "read_from_xz", "read_from_zip"}

    td_iter_method_definitions = get_method_definitions(
        iterDP_file_paths,
        iterDP_files_to_exclude,
        iterDP_deprecated_files,
        "IterDataPipe",
        iterDP_method_to_special_output_type,
        root=str(DATAPIPE_DIR),
    )

    td_iter_method_definitions = [
        s for s in td_iter_method_definitions if all(ex not in s for ex in iter_method_name_exclusion)
    ]

    iter_method_definitions = core_iter_method_definitions + td_iter_method_definitions

    iter_replacements = [("${init_base}", iter_init_base, 0), ("${IterDataPipeMethods}", iter_method_definitions, 4)]

    gen_from_template(
        dir=str(DATAPIPE_DIR),
        template_name="iter/__init__.pyi.in",
        output_name="iter/__init__.pyi",
        replacements=iter_replacements,
    )

    # MapDataPipes
    mapDP_file_paths: List[str] = ["map/load", "map/transform", "map/util"]
    mapDP_files_to_exclude: Set[str] = {"__init__.py"}
    mapDP_deprecated_files: Set[str] = set()
    mapDP_method_to_special_output_type: Dict[str, str] = {
        "unzip": "List[MapDataPipe]",
        "to_iter_datapipe": "IterDataPipe",
    }
    map_method_name_exclusion: Set[str] = set()

    td_map_method_definitions = get_method_definitions(
        mapDP_file_paths,
        mapDP_files_to_exclude,
        mapDP_deprecated_files,
        "MapDataPipe",
        mapDP_method_to_special_output_type,
        root=str(DATAPIPE_DIR),
    )

    td_map_method_definitions = [
        s for s in td_map_method_definitions if all(ex not in s for ex in map_method_name_exclusion)
    ]

    map_method_definitions = core_map_method_definitions + td_map_method_definitions

    map_replacements = [("${init_base}", map_init_base, 0), ("${MapDataPipeMethods}", map_method_definitions, 4)]

    gen_from_template(
        dir=str(DATAPIPE_DIR),
        template_name="map/__init__.pyi.in",
        output_name="map/__init__.pyi",
        replacements=map_replacements,
    )


if __name__ == "__main__":
    gen_pyi()
