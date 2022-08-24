# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains the data pipeline to read from a Paruet and output a DataFrame.
"""

import torcharrow.dtypes as dt
from common import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchdata.datapipes.iter import FileLister, ParquetDataFrameLoader


DTYPE = dt.Struct(
    [dt.Field("label", dt.int64)]
    + [dt.Field(int_name, dt.Float64(nullable=True)) for int_name in DEFAULT_INT_NAMES]
    + [dt.Field(cat_name, dt.Float64(nullable=True)) for cat_name in DEFAULT_CAT_NAMES]
)

source_dp = FileLister(".", masks="*.parquet")
parquet_df_dp = ParquetDataFrameLoader(source_dp, dtype=DTYPE)
