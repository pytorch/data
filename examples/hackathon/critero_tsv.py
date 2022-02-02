# Copyright (c) Facebook, Inc. and its affiliates.
import csv
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torcharrow as ta
import torcharrow.dtypes as dt
from common import CAT_FEATURE_COUNT, DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES, INT_FEATURE_COUNT
from iopath.common.file_io import PathManager, PathManagerFactory
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import Batcher, Mapper

PATH_MANAGER_KEY = "torchrec"
T = TypeVar("T")


def safe_cast(val: T, dest_type: Callable[[T], T], default: T) -> T:
    try:
        return dest_type(val)
    except ValueError:
        return default


DEFAULT_LABEL_NAME = "label"
DEFAULT_COLUMN_NAMES: List[str] = [
    DEFAULT_LABEL_NAME,
    *DEFAULT_INT_NAMES,
    *DEFAULT_CAT_NAMES,
]

COLUMN_TYPE_CASTERS: List[Callable[[Union[int, str]], Union[int, str]]] = [
    lambda val: safe_cast(val, int, 0),
    *(lambda val: safe_cast(val, int, 0) for _ in range(INT_FEATURE_COUNT)),
    *(lambda val: safe_cast(val, str, "") for _ in range(CAT_FEATURE_COUNT)),
]

import torch.utils.data

DTYPE = dt.Struct(
    [
        dt.Field("labels", dt.int8),
        dt.Field(
            "dense_features",
            dt.Struct([dt.Field(int_name, dt.Int32(nullable=True)) for int_name in DEFAULT_INT_NAMES]),
        ),
        dt.Field(
            "sparse_features",
            dt.Struct([dt.Field(cat_name, dt.Int32(nullable=True)) for cat_name in DEFAULT_CAT_NAMES]),
        ),
    ]
)


def _torcharrow_row_mapper(
    row: List[str],
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    label = int(safe_cast(row[0], int, 0))
    dense = tuple(int(safe_cast(row[i], int, 0)) for i in range(1, 1 + INT_FEATURE_COUNT))
    sparse = tuple(
        int(safe_cast(row[i], str, "0") or "0", 16)
        for i in range(1 + INT_FEATURE_COUNT, 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT)
    )
    # TorchArrow doesn't support uint32, but we can save memory
    # by not using int64. Numpy will automatically handle sparse values >= 2 ** 31.
    sparse = tuple(np.array(sparse, dtype=np.int32).tolist())

    return (label, dense, sparse)


def criteo_dataframes_from_tsv(
    paths: Union[str, Iterable[str]],
    *,
    batch_size: int = 128,
) -> IterDataPipe:
    """
    Load Criteo dataset (Kaggle or Terabyte) as TorchArrow DataFrame streams from TSV file(s)

    This implementaiton is inefficient and is used for prototype and test only.

    Args:
        paths (str or Iterable[str]): local paths to TSV files that constitute
            the Kaggle or Criteo 1TB dataset.

    Example:
        >>> datapipe = criteo_dataframes_from_tsv(
        >>>     ["/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv"]
        >>> )
        >>> for df in datapipe:
        >>>    print(df)
    """
    if isinstance(paths, str):
        paths = [paths]

    datapipe = CriteoIterDataPipe(paths, row_mapper=_torcharrow_row_mapper)
    datapipe = Batcher(datapipe, batch_size)
    datapipe = Mapper(datapipe, lambda batch: ta.DataFrame(batch, dtype=DTYPE))

    return datapipe


def _default_row_mapper(example: List[str]) -> Dict[str, Union[int, str]]:
    column_names = reversed(DEFAULT_COLUMN_NAMES)
    column_type_casters = reversed(COLUMN_TYPE_CASTERS)
    return {next(column_names): next(column_type_casters)(val) for val in reversed(example)}


class ReadLinesFromCSV(IterDataPipe):
    def __init__(
        self,
        datapipe: IterDataPipe[Tuple[str, "IOBase"]],
        skip_first_line: bool = False,
        # pyre-ignore[2]
        **kw,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.skip_first_line = skip_first_line
        # pyre-ignore[4]
        self.kw = kw

    def __iter__(self) -> Iterator[List[str]]:
        for _, data in self.datapipe:
            reader = csv.reader(data, **self.kw)
            if self.skip_first_line:
                next(reader, None)
            yield from reader


class LoadFiles(IterDataPipe[Tuple[str, "IOBase"]]):
    """
    Taken and adapted from torch.utils.data.datapipes.iter.LoadFilesFromDisk

    TODO:
    Merge this back or replace this with something in core Datapipes lib
    """

    def __init__(
        self,
        datapipe: Iterable[str],
        mode: str = "b",
        length: int = -1,
        path_manager_key: str = PATH_MANAGER_KEY,
        # pyre-ignore[2]
        **open_kw,
    ) -> None:
        super().__init__()
        self.datapipe: Iterable[str] = datapipe
        self.mode: str = mode
        if self.mode not in ("b", "t", "rb", "rt", "r"):
            raise ValueError(f"Invalid mode {mode}")
        # TODO: enforce typing for each instance based on mode, otherwise
        #       `argument_validation` with this DataPipe may be potentially broken
        self.length: int = length
        # pyre-ignore[4]
        self.open_kw = open_kw
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)
        self.path_manager.set_strict_kwargs_checking(False)

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    # pyre-ignore[3]
    def __iter__(self):
        if self.mode in ("b", "t"):
            self.mode = "r" + self.mode
        for pathname in self.datapipe:
            if not isinstance(pathname, str):
                raise TypeError(f"Expected string type for pathname, but got {type(pathname)}")
            yield (
                pathname,
                self.path_manager.open(pathname, self.mode, **self.open_kw),
            )

    def __len__(self) -> int:
        if self.length == -1:
            raise NotImplementedError
        return self.length


class CriteoIterDataPipe(IterDataPipe):
    """
    IterDataPipe that can be used to stream either the Criteo 1TB Click Logs Dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) or the
    Kaggle/Criteo Display Advertising Dataset
    (https://www.kaggle.com/c/criteo-display-ad-challenge/) from the source TSV
    files.

    Args:
        paths (Iterable[str]): local paths to TSV files that constitute the Criteo
            dataset.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each
            split TSV line.
        open_kw: options to pass to underlying invocation of
            iopath.common.file_io.PathManager.open.

    Example:
        >>> datapipe = CriteoIterDataPipe(
        >>>     ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        >>> )
        >>> datapipe = dp.iter.Batcher(datapipe, 100)
        >>> datapipe = dp.iter.Collator(datapipe)
        >>> batch = next(iter(datapipe))
    """

    def __init__(
        self,
        paths: Iterable[str],
        *,
        # pyre-ignore[2]
        row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
        # pyre-ignore[2]
        **open_kw,
    ) -> None:
        self.paths = paths
        self.row_mapper = row_mapper
        self.open_kw: Any = open_kw  # pyre-ignore[4]

    # pyre-ignore[3]
    def __iter__(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()
        paths = self.paths
        if worker_info is not None:
            paths = (path for (idx, path) in enumerate(paths) if idx % worker_info.num_workers == worker_info.id)
        #  TODO: use these DataPipes instead
        # datapipe = FileLoader(paths, mode="r")
        # datapipe = CSVParser(datapipe, delimiter="\t")
        datapipe = LoadFiles(paths, mode="r", **self.open_kw)
        datapipe = ReadLinesFromCSV(datapipe, delimiter="\t")
        if self.row_mapper:
            datapipe = Mapper(datapipe, self.row_mapper)
        yield from datapipe


# Creating DataFrame from TSV File
dp = criteo_dataframes_from_tsv("day_11_first_3k_rows_original.tsv")
print(list(dp))
