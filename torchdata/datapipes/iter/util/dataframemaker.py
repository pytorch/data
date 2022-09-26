# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Optional, TypeVar

from torch.utils.data.datapipes.utils.common import DILL_AVAILABLE

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

try:  # TODO(637): Create dependency on TorchArrow?
    import pyarrow.parquet as parquet
    import torcharrow
except ImportError:
    torcharrow = None
    parquet = None

if DILL_AVAILABLE:
    import dill

    dill.extend(use_dill=False)

T_co = TypeVar("T_co")


def _construct_dataframe(data, dtype=None, dtype_generator=None, columns=None, device=None):
    if dtype is None:
        dtype = dtype_generator()
    return torcharrow.dataframe(data, dtype=dtype, columns=columns, device=device)


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
        dataframe_size: number of rows of data within each DataFrame, page size can be option
        dtype: specify the `TorchArrow` dtype for the DataFrame, use ``torcharrow.dtypes.DType``
        dtype_generator: function with no input argument that generates a torcharrow.dtypes.DType,
            which overrides dtype if both are given. This is useful for when the desired dtype is
            not serializable.
        columns: List of str that specifies the column names of the DataFrame
        device: specify the device on which the DataFrame will be stored

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> import torcharrow.dtypes as dt
        >>> source_data = [(i,) for i in range(3)]
        >>> source_dp = IterableWrapper(source_data)
        >>> DTYPE = dt.Struct([dt.Field("Values", dt.int32)])
        >>> df_dp = source_dp.dataframe(dtype=DTYPE)
        >>> list(df_dp)[0]
          index    Values
        -------  --------
              0         0
              1         1
              2         2
        dtype: Struct([Field('Values', int32)]), count: 3, null_count: 0
    """

    def __new__(
        cls,
        source_dp: IterDataPipe[T_co],
        dataframe_size: int = 1000,
        dtype=None,
        dtype_generator=None,
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
        df_dp = batch_dp.map(
            partial(_construct_dataframe, dtype=dtype, dtype_generator=dtype_generator, columns=columns, device=device)
        )
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
        dtype: specify the `TorchArrow` dtype for the DataFrame, use ``torcharrow.dtypes.DType``
        device: specify the device on which the DataFrame will be stored

    Example:
        >>> from torchdata.datapipes.iter import FileLister
        >>> import torcharrow.dtypes as dt
        >>> DTYPE = dt.Struct([dt.Field("Values", dt.int32)])
        >>> source_dp = FileLister(".", masks="df*.parquet")
        >>> parquet_df_dp = source_dp.load_parquet_as_df(dtype=DTYPE)
        >>> list(parquet_df_dp)[0]
          index    Values
        -------  --------
              0         0
              1         1
              2         2
        dtype: Struct([Field('Values', int32)]), count: 3, null_count: 0
    """

    def __init__(
        self,
        source_dp: IterDataPipe[str],
        dtype=None,
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
                # TODO(638): More fine-grain control over the number of rows or row group per DataFrame
                row_group = parquet_file.read_row_group(i, columns=self.columns, use_threads=self.use_threads)
                yield torcharrow.from_arrow(row_group, dtype=self.dtype)

    def __getstate__(self):
        if DILL_AVAILABLE:
            dill_dtype = dill.dumps(self.dtype)
        else:
            dill_dtype = self.dtype
        state = (self.source_dp, dill_dtype, self.columns, self.device, self.use_threads)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (self.source_dp, dill_dtype, self.columns, self.device, self.use_threads) = state
        if DILL_AVAILABLE:
            self.dtype = dill.loads(dill_dtype)  # type: ignore[assignment]
        else:
            self.dtype = dill_dtype  # type: ignore[assignment]
