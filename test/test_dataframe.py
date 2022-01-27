# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest
import warnings

import expecttest

from _utils._common_utils_for_test import create_temp_dir, reset_after_n_next_calls

from torchdata.datapipes.iter import DataFrameMaker, FileLister, FileOpener, IterableWrapper

try:
    import torcharrow
    import torcharrow.dtypes as dt

    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
skipIfNoArrow = unittest.skipIf(not HAS_ARROW, "no TorchArrow or PyArrow.")


@skipIfNoArrow
class TestDataFrame(expecttest.TestCase):
    def setUp(self):
        self.temp_dir = create_temp_dir()

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(f"TestDataFrame was not able to cleanup temp dir due to {e}")

    def _custom_files_set_up(self, files):
        for fname, content in files.items():
            temp_file_path = os.path.join(self.temp_dir.name, fname)
            with open(temp_file_path, "w") as f:
                f.write(content)

    def _compare_dataframes(self, expected_df, actual_df):
        for exp, act in zip(expected_df, actual_df):
            self.assertEqual(exp, act)

    def test_dataframe_maker_iterdatapipe(self):

        source_data = [(i,) for i in range(10)]
        source_dp = IterableWrapper(source_data)
        DTYPE = dt.Struct([dt.Field("Values", dt.int32)])

        # Functional Test: DataPipe correctly converts into a single TorchArrow DataFrame
        df_dp = source_dp.dataframe(dtype=DTYPE)
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
        df_dp = source_dp.dataframe(dtype=DTYPE)
        self.assertEqual(1, len(df_dp))
        self.assertEqual(10, len(list(df_dp)[0]))
        df_dp = source_dp.dataframe(dataframe_size=5, dtype=DTYPE)
        self.assertEqual(2, len(df_dp))
        self.assertEqual(5, len(list(df_dp)[0]))

        # Reset Test:
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(df_dp, n_elements_before_reset)
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

        DTYPE = dt.Struct([dt.Field("key", dt.string), dt.Field("item", dt.string)])
        df_dp = csv_dict_parser_dp.dataframe(dtype=DTYPE, columns=["key", "item"])
        expected_dfs = [torcharrow.DataFrame([{"key": "a", "item": "1"}, {"key": "b", "item": "2"}], dtype=DTYPE)]
        for exp_df, act_df in zip(expected_dfs, list(df_dp)):
            self._compare_dataframes(exp_df, act_df)
