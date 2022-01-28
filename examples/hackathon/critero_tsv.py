# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains the data pipeline to read from a TSV file and output a DataFrame.
"""
import os

import torcharrow.dtypes as dt
from common import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchdata.datapipes.iter import FileLister, FileOpener


# Helper Functions
def get_name(path_and_stream):
    return os.path.basename(path_and_stream[0]), path_and_stream[1]


# Convert TSV Strings to Float
def str_to_num(s):
    try:
        return float(s)
    except Exception:
        return float("NaN")


# Process each row
def row_pre_process(row):
    row = row[1:]  # Remove index column from TSV
    for i in range(len(row)):
        row[i] = str_to_num(row[i])
    return tuple(row)


DTYPE = dt.Struct(
    [dt.Field("label", dt.float64)]
    + [dt.Field(int_name, dt.Float64(nullable=True)) for int_name in DEFAULT_INT_NAMES]
    + [dt.Field(cat_name, dt.Float64(nullable=True)) for cat_name in DEFAULT_CAT_NAMES]
)

source_dp = FileLister(".", masks="day_11_first_3k_rows.tsv")
file_dp = FileOpener(source_dp, mode="b")
map_dp = file_dp.map(get_name)
csv_parser_dp = map_dp.parse_csv(delimiter="\t", skip_lines=1)
columns_names = list(map_dp.parse_csv(delimiter="\t"))[0][1:]
processed_dp = csv_parser_dp.map(row_pre_process)
header_dp = processed_dp.header(1000)  # limit to 1000 rows to save time
# TODO: The follow operation is very slow due to torcharrow.DataFrame
#       It gives the following warning:
#       "UserWarning: append for type NumericalColumnCpu is suported only with prototype implementation,
#        which may result in degenerated performance"
dfs = header_dp.dataframe(dataframe_size=1000, dtype=DTYPE)
print(list(dfs))
