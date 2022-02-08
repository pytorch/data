# Copyright (c) Facebook, Inc. and its affiliates.
from functools import partial
from typing import List, Optional, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

try:  # TODO: Create dependency on TorchArrow?
    import pyarrow.parquet as parquet
    import torcharrow
except ImportError:
    torcharrow = None
    parquet = None

T_co = TypeVar("T_co")


@functional_datapipe("dataframe")
class DataFrameMakerIterDataPipe(IterDataPipe):  # IterDataPipe[torcharrow.IDataFrame[T_co]]
    r"""
    Takes rows of data, batches a number of them together and creates `TorchArrow`
    DataFrames (functional name: ``dataframe``).

    Note:
        There is a trade-off between having a large number of rows within a DataFrame and usage of memory. Please
        choose a value carefully.

    Args:
        source_dp: IterDataPipe containing rows of data
        dataframe_size: number of rows of data within each DataFrame
        dtype: specify the `TorchArrow` dtype for the DataFrame
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
        if torcharrow is None:
            raise ImportError(
                "The library 'torcharrow' is necessary for this DataPipe but it is not available."
                "Please visit https://github.com/facebookresearch/torcharrow/ to install it."
            )
        # In this version, DF tracing is not available, which would allow DataPipe to run DataFrame operations
        batch_dp = source_dp.batch(dataframe_size)
        df_dp = batch_dp.map(partial(torcharrow.DataFrame, dtype=dtype, columns=columns, device=device))
        return df_dp


@functional_datapipe("load_parquet_as_df")
class ParquetDFLoaderIterDataPipe(IterDataPipe):  # IterDataPipe[torcharrow.IDataFrame[T_co]]
    r"""
    Takes in paths to Parquet files and return a `TorchArrow` DataFrame for each row group
    within a Parquet file (functional name: ``load_parquet_as_df``).

    Args:
        source_dp: source DataPipe containing paths to the Parquet files
        columns: List of `str` that specifies the column names of the DataFrame
        use_threads: if ``True``, Parquet reader will perform multi-threaded column reads
        dtype: specify the `TorchArrow` dtype for the DataFrame
        device: specify the device on which the DataFrame will be stored
    """

    def __init__(
        self,
        source_dp: IterDataPipe[str],
        dtype=None,  # Optional[torcharrow.dtypes.DType]
        columns: Optional[List[str]] = None,
        device: str = "",
        use_threads: bool = False,
    ):
        if torcharrow is None:
            raise ImportError(
                "The library 'torcharrow' is necessary for this DataPipe but it is not available."
                "Please visit https://github.com/facebookresearch/torcharrow/ to install it."
            )
        if parquet is None:
            raise ImportError("The library 'parquet' is necessary for this DataPipe but it is not available.")
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
                # TODO: More fine-grain control over the number of rows or row group per DataFrame
                row_group = parquet_file.read_row_group(i, columns=self.columns, use_threads=self.use_threads)
                yield torcharrow.from_arrow(row_group, dtype=self.dtype)
