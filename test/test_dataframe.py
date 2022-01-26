# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest
import warnings

import expecttest
from _utils._common_utils_for_test import create_temp_dir, reset_after_n_next_calls

from torchdata.datapipes.iter import DataFrameMaker, FileLister, IterableWrapper, ParquetDFReader

try:
    import pyarrow
    import pyarrow.parquet as parquet
    import torcharrow
    import torcharrow.dtypes as dt

    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
skipIfNoArrow = unittest.skipIf(not HAS_ARROW, "no TorchArrow or PyArrow.")


class TestDataFrame(expecttest.TestCase):
    def _compare_dataframes(self, expected_df, actual_df):
        for exp, act in zip(expected_df, actual_df):
            self.assertEqual(exp, act)

    def _write_df_as_parquet(self, df: torcharrow.IDataFrame, fname: str) -> None:
        table = df.to_arrow()
        parquet.write_table(table, os.path.join(self.temp_dir.name, fname))

    @skipIfNoArrow
    def setUp(self) -> None:
        self.temp_dir = create_temp_dir()

        # Create TorchArrow DataFrames
        DTYPE = dt.Struct([dt.Field("Values", dt.int32)])
        df1 = torcharrow.DataFrame([(i,) for i in range(10)], dtype=DTYPE)
        df2 = torcharrow.DataFrame([(i,) for i in range(100)], dtype=DTYPE)

        # Write them as parquet files
        for i, df in enumerate([df1, df2]):
            fname = f"df{i}.parquet"
            self._write_df_as_parquet(df, fname)

    @skipIfNoArrow
    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(f"TestDataFrame was not able to cleanup temp dir due to {e}")

    @skipIfNoArrow
    def test_dataframe_maker_iterdatapipe(self):

        source_data = [(i,) for i in range(10)]
        source_dp = IterableWrapper(source_data)
        DTYPE = dt.Struct([dt.Field("Values", dt.int32)])

        # Functional Test: DataPipe correctly converts into a single TorchArrow DataFrame
        df_dp = source_dp.convert_to_dataframe(dtype=DTYPE)
        df = list(df_dp)[0]
        expected_df = torcharrow.DataFrame([(i,) for i in range(10)], dtype=DTYPE)
        self._compare_dataframes(expected_df, df)

        # Functional Test: DataPipe correctly converts into multiple TorchArrow DataFrames, based on size argument
        df_dp = DataFrameMaker(source_dp, dataframe_size=5, dtype=DTYPE)
        dfs = list(df_dp)
        expected_dfs = [
            torcharrow.DataFrame([(i,) for i in range(5)], dtype=DTYPE),
            torcharrow.DataFrame([(i,) for i in range(5, 10)], dtype=DTYPE),
        ]
        for exp_df, act_df in zip(expected_dfs, dfs):
            self._compare_dataframes(exp_df, act_df)

        # __len__ Test:
        df_dp = source_dp.convert_to_dataframe(dtype=DTYPE)
        self.assertEqual(1, len(df_dp))
        self.assertEqual(10, len(list(df_dp)[0]))
        df_dp = source_dp.convert_to_dataframe(dataframe_size=5, dtype=DTYPE)
        self.assertEqual(2, len(df_dp))
        self.assertEqual(5, len(list(df_dp)[0]))

        # Reset Test:
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(df_dp, n_elements_before_reset)
        for exp_df, act_df in zip(expected_dfs[:1], res_before_reset):
            self._compare_dataframes(exp_df, act_df)
        for exp_df, act_df in zip(expected_dfs, res_after_reset):
            self._compare_dataframes(exp_df, act_df)

    @skipIfNoArrow
    def test_parquet_dataframe_reader_iterdatapipe(self):

        DTYPE = dt.Struct([dt.Field("Values", dt.int32)])

        # Functional Test: Parquet
        source_dp = FileLister(self.temp_dir.name)
        parquet_df_dp = ParquetDFReader(source_dp, dtype=DTYPE)
        expected_dfs = [
            torcharrow.DataFrame([(i,) for i in range(10)], dtype=DTYPE),
            torcharrow.DataFrame([(i,) for i in range(100)], dtype=DTYPE),
        ]
        for exp_df, act_df in zip(expected_dfs, list(parquet_df_dp)):
            self._compare_dataframes(exp_df, act_df)

        # __len__ Test: inherits length from the source_dp
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length"):
            self.assertEqual(2, len(parquet_df_dp))

        # Reset Test:
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(parquet_df_dp, n_elements_before_reset)
        for exp_df, act_df in zip(expected_dfs[:1], res_before_reset):
            self._compare_dataframes(exp_df, act_df)
        for exp_df, act_df in zip(expected_dfs, res_after_reset):
            self._compare_dataframes(exp_df, act_df)
