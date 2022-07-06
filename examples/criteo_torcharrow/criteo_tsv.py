# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains the data pipeline to read from a TSV file and output a DataFrame.
"""
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torcharrow as ta

import torcharrow.dtypes as dt
import torcharrow.pytorch as tap
import torcharrow_wrapper  # noqa: F401
from common import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_COLUMN_NAMES,
    DEFAULT_INT_NAMES,
    INT_FEATURE_COUNT,
    safe_cast,
)
from iopath.common.file_io import PathManagerFactory
from torch.utils.data import get_worker_info

from torch.utils.data.datapipes.dataframe.dataframes import CaptureLikeMock
from torcharrow import functional
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import Batcher, CSVParser, IoPathFileOpener, IterableWrapper, IterDataPipe, Mapper

PATH_MANAGER_KEY = "torchrec"
T = TypeVar("T")


COLUMN_TYPE_CASTERS: List[Callable[[Union[int, str]], Union[int, str]]] = [
    lambda val: safe_cast(val, int, 0),
    *(lambda val: safe_cast(val, int, 0) for _ in range(INT_FEATURE_COUNT)),
    *(lambda val: safe_cast(val, str, "") for _ in range(CAT_FEATURE_COUNT)),
]

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


def _torcharrow_row_mapper(row: List[str]) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    label = int(safe_cast(row[0], int, 0))
    dense = tuple(int(safe_cast(row[i], int, 0)) for i in range(1, 1 + INT_FEATURE_COUNT))
    sparse = tuple(
        int(safe_cast(row[i], str, "0") or "0", 16)
        for i in range(1 + INT_FEATURE_COUNT, 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT)
    )
    # TorchArrow doesn't support uint32, but we can save memory
    # by not using int64. Numpy will automatically handle sparse values >= 2 ** 31.
    sparse = tuple(np.array(sparse, dtype=np.int32).tolist())
    return label, dense, sparse


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
        batch_size (int): number of rows within each DataFrame

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
    datapipe = Mapper(datapipe, lambda batch: ta.dataframe(batch, dtype=DTYPE))
    return datapipe.trace_as_dataframe()


def _default_row_mapper(example: List[str]) -> Dict[str, Union[int, str]]:
    column_names = reversed(DEFAULT_COLUMN_NAMES)
    column_type_casters = reversed(COLUMN_TYPE_CASTERS)
    return {next(column_names): next(column_type_casters)(val) for val in reversed(example)}


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
    ) -> None:
        self.paths = paths
        self.row_mapper = row_mapper

    # pyre-ignore[3]
    def __iter__(self) -> Iterator[Any]:
        worker_info = get_worker_info()
        paths = self.paths
        if worker_info is not None:
            paths = (path for (idx, path) in enumerate(paths) if idx % worker_info.num_workers == worker_info.id)
        paths = IterableWrapper(paths)
        datapipe = IoPathFileOpener(paths, mode="r", pathmgr=PathManagerFactory().get(PATH_MANAGER_KEY))
        datapipe = CSVParser(datapipe, delimiter="\t")
        if self.row_mapper:
            datapipe = Mapper(datapipe, self.row_mapper)
        yield from datapipe


# Creating DataFrame from TSV File
df = criteo_dataframes_from_tsv("day_11_first_3k_rows_original.tsv")

# TODO(VitalyFedyunin): Optimize this operation
df = df.shuffle()

df["dense_features"] = df["dense_features"].fill_null(0)
df["sparse_features"] = df["sparse_features"].fill_null(0)

# This Mock going to to removed as hackathon followup, when torcharrow.functional will
# accept StreamDataFrame
with CaptureLikeMock("torcharrow.functional.array_constructor"):
    for field in df["sparse_features"].columns:
        df["sparse_features"][field] = functional.array_constructor(df["sparse_features"][field])

df["dense_features"] = (df["dense_features"] + 3).log()
df["labels"] = df["labels"].cast(dt.int32)

df = df.batch(10)

conversion = {
    "dense_features": tap.rec.Dense(),
    "sparse_features": tap.rec.Dense(),  # Sparse not implemented yet in torcharrow
    # Because labels are unlisted it works like "labels": tap.rec.Default(),
}
df = df.collate(conversion=conversion)

reading_service = MultiProcessingReadingService(num_workers=0)

dl = DataLoader2(df, reading_service=reading_service)

print("Iterating DataLoader now")

for item in dl:
    labels, dense_features, sparse_features = item
    print(labels)
    print(dense_features)
    print(sparse_features)
    break
