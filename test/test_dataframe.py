# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import warnings
from itertools import chain

import expecttest
from _utils._common_utils_for_test import create_temp_dir, reset_after_n_next_calls
from torchdata.datapipes.iter import (
    DataFrameMaker,
    FileLister,
    FileOpener,
    IterableWrapper,
    ParquetDataFrameLoader,
)

try:
    import torcharrow
    import torcharrow.dtypes as dt

    HAS_TORCHARROW = True
except ImportError:
    HAS_TORCHARROW = False

try:
    import pyarrow
    import pyarrow.parquet as parquet

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
skipIfNoPyArrow = unittest.skipIf(not HAS_PYARROW, "no PyArrow.")
skipIfNoTorchArrow = unittest.skipIf(not HAS_TORCHARROW, "no TorchArrow.")


@skipIfNoTorchArrow
class TestDataFrame(expecttest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = create_temp_dir()
        if HAS_PYARROW:
            self._write_parquet_files()

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(f"TestDataFrame was not able to cleanup temp dir due to {e}")

    def _write_parquet_files(self):
        # Create TorchArrow DataFrames
        DTYPE = dt.Struct([dt.Field("Values", dt.int32)])
        df1 = torcharrow.dataframe([(i,) for i in range(10)], dtype=DTYPE)
        df2 = torcharrow.dataframe([(i,) for i in range(100)], dtype=DTYPE)
        # Write them as parquet files
        for i, df in enumerate([df1, df2]):
            fname = f"df{i}.parquet"
            self._write_df_as_parquet(df, fname)
        self._write_multiple_dfs_as_parquest([df1, df2], fname="merged.parquet")

    def _custom_files_set_up(self, files):
        for fname, content in files.items():
            temp_file_path = os.path.join(self.temp_dir.name, fname)
            with open(temp_file_path, "w") as f:
                f.write(content)

    def _compare_dataframes(self, expected_df, actual_df):
        self.assertEqual(len(expected_df), len(actual_df))
        for exp, act in zip(expected_df, actual_df):
            self.assertEqual(exp, act)

    def _write_df_as_parquet(self, df, fname: str) -> None:
        table = df.to_arrow()
        parquet.write_table(table, os.path.join(self.temp_dir.name, fname))

    def _write_multiple_dfs_as_parquest(self, dfs, fname: str) -> None:
        tables = [df.to_arrow() for df in dfs]
        merged_table = pyarrow.concat_tables(tables)
        parquet.write_table(merged_table, os.path.join(self.temp_dir.name, fname))

    def test_dataframe_maker_iterdatapipe(self):
        source_data = [(i,) for i in range(10)]
        source_dp = IterableWrapper(source_data)
        DTYPE = dt.Struct([dt.Field("Values", dt.int32)])

        # Functional Test: DataPipe correctly converts into a single TorchArrow DataFrame
        df_dp = source_dp.dataframe(dtype=DTYPE)
        df = list(df_dp)[0]
        expected_df = torcharrow.dataframe([(i,) for i in range(10)], dtype=DTYPE)
        self._compare_dataframes(expected_df, df)

        # Functional Test: DataPipe correctly converts into multiple TorchArrow DataFrames, based on size argument
        df_dp = DataFrameMaker(source_dp, dataframe_size=5, dtype=DTYPE)
        dfs = list(df_dp)
        expected_dfs = [
            torcharrow.dataframe([(i,) for i in range(5)], dtype=DTYPE),
            torcharrow.dataframe([(i,) for i in range(5, 10)], dtype=DTYPE),
        ]
        for exp_df, act_df in zip(expected_dfs, dfs):
            self._compare_dataframes(exp_df, act_df)

        # __len__ Test:
        df_dp = source_dp.dataframe(dtype=DTYPE)
        self.assertEqual(1, len(df_dp))
        self.assertEqual(10, len(list(df_dp)[0]))
        df_dp = source_dp.dataframe(dataframe_size=5, dtype=DTYPE)
        self.assertEqual(2, len(df_dp))
        self.assertEqual(5, len(list(df_dp)[0]))

        # Reset Test:
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            df_dp, n_elements_before_reset
        )
        for exp_df, act_df in zip(expected_dfs[:1], res_before_reset):
            self._compare_dataframes(exp_df, act_df)
        for exp_df, act_df in zip(expected_dfs, res_after_reset):
            self._compare_dataframes(exp_df, act_df)

    def test_dataframe_maker_with_csv(self):
        def get_name(path_and_stream):
            return os.path.basename(path_and_stream[0]), path_and_stream[1]

        csv_files = {"1.csv": "key,item\na,1\nb,2"}
        self._custom_files_set_up(csv_files)
        datapipe1 = FileLister(self.temp_dir.name, "*.csv")
        datapipe2 = FileOpener(datapipe1, mode="b")
        datapipe3 = datapipe2.map(get_name)
        csv_dict_parser_dp = datapipe3.parse_csv_as_dict()

        # Functional Test: Correctly generate TorchArrow DataFrame from CSV
        DTYPE = dt.Struct([dt.Field("key", dt.string), dt.Field("item", dt.string)])
        df_dp = csv_dict_parser_dp.dataframe(dtype=DTYPE, columns=["key", "item"])
        expected_dfs = [
            torcharrow.dataframe(
                [{"key": "a", "item": "1"}, {"key": "b", "item": "2"}], dtype=DTYPE
            )
        ]
        for exp_df, act_df in zip(expected_dfs, list(df_dp)):
            self._compare_dataframes(exp_df, act_df)

        # Functional: making sure DataPipe works even without `columns` input
        df_dp = csv_dict_parser_dp.dataframe(dtype=DTYPE)
        for exp_df, act_df in zip(expected_dfs, list(df_dp)):
            self._compare_dataframes(exp_df, act_df)

    @skipIfNoPyArrow
    def test_parquet_dataframe_reader_iterdatapipe(self):
        DTYPE = dt.Struct([dt.Field("Values", dt.int32)])

        # Functional Test: read from Parquet files and output TorchArrow DataFrames
        source_dp = FileLister(self.temp_dir.name, masks="df*.parquet")
        parquet_df_dp = ParquetDataFrameLoader(source_dp, dtype=DTYPE)
        expected_dfs = [
            torcharrow.dataframe([(i,) for i in range(10)], dtype=DTYPE),
            torcharrow.dataframe([(i,) for i in range(100)], dtype=DTYPE),
        ]
        for exp_df, act_df in zip(expected_dfs, list(parquet_df_dp)):
            self._compare_dataframes(exp_df, act_df)

        # Functional Test: correctly read from a Parquet file that was a merged DataFrame
        merged_source_dp = FileLister(self.temp_dir.name, masks="merged.parquet")
        merged_parquet_df_dp = ParquetDataFrameLoader(merged_source_dp, dtype=DTYPE)
        expected_merged_dfs = [
            torcharrow.dataframe(
                [(i,) for i in chain(range(10), range(100))], dtype=DTYPE
            )
        ]
        for exp_df, act_df in zip(expected_merged_dfs, list(merged_parquet_df_dp)):
            self._compare_dataframes(exp_df, act_df)

        # __len__ Test: no valid length because we do not know the number of row groups in advance
        with self.assertRaisesRegex(TypeError, "has no len"):
            len(parquet_df_dp)

        # Reset Test:
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            parquet_df_dp, n_elements_before_reset
        )
        for exp_df, act_df in zip(expected_dfs[:1], res_before_reset):
            self._compare_dataframes(exp_df, act_df)
        for exp_df, act_df in zip(expected_dfs, res_after_reset):
            self._compare_dataframes(exp_df, act_df)


if __name__ == "__main__":
    unittest.main()
