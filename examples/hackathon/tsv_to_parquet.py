# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file pre-process the source file and save it as a TSV file and a Parquet file.
You do not need to re-run this file if "day_11_first_3k_rows.parquet" and "day_11_first_3k_rows.tsv" exist locally
"""
import pandas
import pyarrow
import pyarrow.parquet as parquet
from common import DEFAULT_CAT_NAMES, DEFAULT_COLUMN_NAMES, safe_hex_to_int


# Read TSV File with Pandas
tsv_fname = "day_11_first_3k_rows_original.tsv"
df = pandas.read_csv(tsv_fname, sep="\t")
df.columns = DEFAULT_COLUMN_NAMES

# Convert hex strings to interger
for i, row in df.iterrows():
    for cat_col in DEFAULT_CAT_NAMES:
        df.at[i, cat_col] = safe_hex_to_int(row[cat_col])


# Convert to PyArrow table and write to disk as parquet file
table = pyarrow.Table.from_pandas(df=df)
parquet_fname = "day_11_first_3k_rows.parquet"
parquet.write_table(table, parquet_fname)

# Write to a new .tsv file
df.to_csv("day_11_first_3k_rows.tsv", sep="\t")
