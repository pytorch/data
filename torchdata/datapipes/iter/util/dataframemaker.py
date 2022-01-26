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
class DataFrameMakerIterDataPipe(IterDataPipe[torcharrow.IDataFrame[T_co]]):
    r"""
    Iterable DataPipe that takes rows of data, batch a number of them together and create TorchArrow DataFrames.
    Note that there is a trade-off between having a large number of rows within a DataFrame and usage of memory.

    Args:
        source_dp: IterDataPipe containing rows of data
        dataframe_size: number of rows of data within each DataFrame
        dtype: specify the TorchArrow dtype (torcharrow.dtypes.DType) for the DataFrame
        columns: List of str that specifies the column names of the DataFrame
        device: specify the device on which the DataFrame will be stored
    """

    def __new__(
        cls,
        source_dp: IterDataPipe[T_co],
        dataframe_size: int = 1000,  # or Page Size
        dtype=None,
        columns: Optional[List[str]] = None,
        device: str = "",
    ):
        # In this version, DF tracing is not available, which would allow DataPipe to run DataFrame operations
        batch_dp = source_dp.batch(dataframe_size)
        df_dp = batch_dp.map(partial(torcharrow.DataFrame, dtype=dtype, columns=columns, device=device))
        return df_dp


@functional_datapipe("load_parquet_as_df")
class ParquetDFIterDataPipe(IterDataPipe):
    r"""
    Iterable DataPipe that takes in paths to Parquet files and return a TorchArrow DataFrame for each Parquet file.

    Args:
        source_dp: source DataPipe containing paths to the Parquet files
        columns: List of str that specifies the column names of the DataFrame
        use_threads: if True, Parquet reader will perform multi-threaded column reads
        dtype: specify the TorchArrow dtype for the DataFrame
        device: specify the device on which the DataFrame will be stored
    """

    def __new__(
        cls,
        source_dp: IterDataPipe[str],
        columns: Optional[List[str]] = None,
        use_threads: bool = False,
        dtype=None,
        device: str = "",
    ):
        table_dp = source_dp.map(partial(parquet.read_table, columns=columns, use_threads=use_threads))
        df_dp = table_dp.map(torcharrow.from_arrow)
        return df_dp
