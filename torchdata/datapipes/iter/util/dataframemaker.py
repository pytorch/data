# Copyright (c) Facebook, Inc. and its affiliates.
from functools import partial
from typing import List, Optional, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

try:  # TODO: Create dependency on TorchArrow?
    import torcharrow
except ImportError:
    torcharrow = None

T_co = TypeVar("T_co")


@functional_datapipe("dataframe")
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
        if torcharrow is None:
            raise ImportError(
                "The library 'torcharrow' is necessary for this DataPipe but it is not available."
                "Please visit https://github.com/facebookresearch/torcharrow/ to install it."
            )
        # In this version, DF tracing is not available, which would allow DataPipe to run DataFrame operations
        batch_dp = source_dp.batch(dataframe_size)
        df_dp = batch_dp.map(partial(torcharrow.DataFrame, dtype=dtype, columns=columns, device=device))
        return df_dp
