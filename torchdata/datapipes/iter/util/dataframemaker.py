# Copyright (c) Facebook, Inc. and its affiliates.
from functools import partial
from typing import List, Optional, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

try:
    import pyarrow.parquet as parquet
    import torcharrow
except ImportError:
    torcharrow = None
    parquet = None

T_co = TypeVar("T_co")


@functional_datapipe("convert_to_dataframe")
class DataFrameMakerIterDataPipe(IterDataPipe):  # IterDataPipe[torcharrow.IDataFrame[T_co]]
    r"""
    Iterable DataPipe that takes rows of data, batch a number of them together and create TorchArrow DataFrames.
    Note that there is a trade-off between having a large number of rows within a DataFrame and usage of memory.

    Args:
        source_dp: IterDataPipe containing rows of data
        dataframe_size: number of rows of data within each DataFrame
        dtype: specify the TorchArrow dtype for the DataFrame
        columns: List of str that specifies the column names of the DataFrame
        device: specify the device on which the DataFrame will be stored
    """

    def __new__(
        cls,
        source_dp: IterDataPipe[T_co],
        dataframe_size: int = 1000,  # or Page Size
        dtype=None,  # Optional[torcharrow.dtypes.DType]
        columns: Optional[List[str]] = None,
        device: str = "",
    ):
        # In this version, DF tracing is not available, which would allow DataPipe to run DataFrame operations
        batch_dp = source_dp.batch(dataframe_size)
        df_dp = batch_dp.map(partial(torcharrow.DataFrame, dtype=dtype, columns=columns, device=device))
        return df_dp


@functional_datapipe("load_parquet_as_df")
class ParquetDFIterDataPipe(IterDataPipe):  # IterDataPipe[torcharrow.IDataFrame[T_co]]
    r"""
    Iterable DataPipe that takes in paths to Parquet files and return a TorchArrow DataFrame for each row group
    within a Parquet file.

    Args:
        source_dp: source DataPipe containing paths to the Parquet files
        columns: List of str that specifies the column names of the DataFrame
        use_threads: if True, Parquet reader will perform multi-threaded column reads
        dtype: specify the TorchArrow dtype for the DataFrame
        device: specify the device on which the DataFrame will be stored
    """

    def __init__(
        self,
        source_dp: IterDataPipe[str],
        columns: Optional[List[str]] = None,
        use_threads: bool = False,
        dtype=None,  # Optional[torcharrow.dtypes.DType]
        device: str = "",
    ):
        self.source_dp = source_dp
        self.columns = columns
        self.use_threads = use_threads
        self.dtype = dtype
        self.device = device

    def __iter__(self):
        for path in self.source_dp:
            parquet_file = parquet.ParquetFile(path)
            num_row_groups = parquet_file.num_row_groups
            for i in range(num_row_groups):
                row_group = parquet_file.read_row_group(i, columns=self.columns, use_threads=self.use_threads)
                yield torcharrow.from_arrow(row_group)
