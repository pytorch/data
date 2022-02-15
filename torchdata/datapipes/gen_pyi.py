import pathlib
from typing import Dict, List, Optional, Set

from torch.utils.data.gen_pyi import FileManager, get_method_definitions


def get_lines_base_file(base_file_path: str, to_skip: Optional[Set[str]] = None):
    with open(base_file_path) as f:
        lines = f.readlines()
        res = []
        for line in lines:
            for skip_line in to_skip:
                if skip_line not in line:
                    res.append(line)
        return res


def main() -> None:

    iter_init_base = get_lines_base_file("iter/__init__.py", {"from torch.utils.data import IterDataPipe"})

    iterDP_file_paths: List[str] = ["iter/load", "iter/transform", "iter/util"]
    iterDP_files_to_exclude: Set[str] = {"__init__.py"}
    iterDP_deprecated_files: Set[str] = set()
    iterDP_method_to_special_output_type: Dict[str, str] = {
        "bucketbatch": "IterDataPipe",
        "dataframe": "torcharrow.DataFrame",
        "end_caching": "IterDataPipe",
        "unzip": "List[IterDataPipe]",
    }

    iter_method_definitions = get_method_definitions(
        iterDP_file_paths,
        iterDP_files_to_exclude,
        iterDP_deprecated_files,
        "IterDataPipe",
        iterDP_method_to_special_output_type,
        root=str(pathlib.Path(__file__).parent.resolve()),
    )

    fm = FileManager(install_dir=".", template_dir=".", dry_run=False)
    fm.write_with_template(
        filename="iter/__init__.pyi",
        template_fn="iter/__init__.pyi.in",
        env_callable=lambda: {"init_base": iter_init_base, "IterDataPipeMethods": iter_method_definitions},
    )
    # TODO: Add map_method_definitions when there are MapDataPipes defined in this library


if __name__ == "__main__":
    main()  # TODO: Run this script automatically within the build and CI process
