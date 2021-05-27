import operator
import unittest

import torcharrow.dtypes as dt
from torcharrow import IListColumn, INumericalColumn, Scope

from .test_list_column import TestListColumn


class TestListColumnStd(TestListColumn):
    def setUp(self):
        self.ts = Scope({"device": "std"})

    def test_empty(self):
        c = self.ts.Column(dt.List(dt.int64))

        self.assertTrue(isinstance(c._data, INumericalColumn))
        self.assertEqual(c._data.dtype, dt.int64)

        self.assertEqual(len(c._offsets), 1)
        self.assertEqual(c._offsets[0], 0)
        self.assertEqual(len(c._mask), 0)

        self.base_test_empty()

    def test_nonempty(self):
        self.base_test_nonempty()

    def test_nested_numerical_twice(self):
        self.base_test_nested_numerical_twice()

    def test_nested_string_once(self):
        self.base_test_nested_string_once()

    def test_nested_string_twice(self):
        self.base_test_nested_string_twice()

    def test_get_count_join(self):
        self.base_test_get_count_join()

    def test_map_reduce_etc(self):
        self.base_test_map_reduce_etc()


if __name__ == "__main__":
    unittest.main()
