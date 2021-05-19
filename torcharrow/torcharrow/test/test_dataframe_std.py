import functools
import statistics
import unittest

import torcharrow.dtypes as dt
from torcharrow import IDataFrame, Scope, me
from .test_dataframe import TestDataFrame
# run python3 -m unittest outside this directory to run all tests


class TestDataFrameStd(TestDataFrame):
    def setUp(self):
        self.ts = Scope({"device": "std"})

    def test_internals_empty(self):
    	return self.base_test_internals_empty()

    def test_internals_full(self):
    	return self.base_test_internals_full()

    def test_internals_full_nullable(self):
    	return self.base_test_internals_full_nullable()

    def test_internals_column_indexing(self):
    	return self.base_test_internals_column_indexing()

    def test_infer(self):
    	return self.base_test_infer()

    def test_map_where_filter(self):
    	return self.base_test_map_where_filter()

    def test_sort_stuff(self):
    	return self.base_test_sort_stuff()

    def test_operators(self):
    	return self.base_test_operators()

    def test_na_handling(self):
    	return self.base_test_na_handling()

    def test_agg_handling(self):
    	return self.base_test_agg_handling()

    def test_isin(self):
    	return self.base_test_isin()

    def test_isin2(self):
    	return self.base_test_isin2()

    def test_describe_dataframe(self):
    	return self.base_test_describe_dataframe()

    def test_drop_keep_rename_reorder_pipe(self):
    	return self.base_test_drop_keep_rename_reorder_pipe()

    def test_me_on_str(self):
    	return self.base_test_me_on_str()

    def test_locals_and_me_equivalence(self):
    	return self.base_test_locals_and_me_equivalence()

    def test_groupby_size_pipe(self):
    	return self.base_test_groupby_size_pipe()

    def test_groupby_agg(self):
    	return self.base_test_groupby_agg()

    def test_groupby_iter_get_item_ops(self):
    	return self.base_test_groupby_iter_get_item_ops()


if __name__ == "__main__":
    unittest.main()
