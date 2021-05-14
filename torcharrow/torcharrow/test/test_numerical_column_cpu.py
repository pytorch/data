import unittest

import torcharrow.dtypes as dt
from torcharrow import INumericalColumn
from torcharrow import Scope

from .test_numerical_column import TestNumericalColumn


class TestNumericalColumnCpu(TestNumericalColumn):
    def setUp(self):
        self.ts = Scope({"device": "cpu"})

    def test_internals_empty(self):
        empty_i64_column = self.ts.Column(dtype=dt.int64)

        # testing internals...
        self.assertTrue(isinstance(empty_i64_column, INumericalColumn))
        self.assertEqual(empty_i64_column.dtype, dt.int64)
        self.assertEqual(len(empty_i64_column), 0)
        self.assertEqual(empty_i64_column.null_count(), 0)
        self.assertEqual(len(empty_i64_column), 0)
        self.assertEqual(empty_i64_column.to, "cpu")

    def test_internals_full(self):
        col = self.ts.Column([i for i in range(4)], dtype=dt.int64)

        # self.assertEqual(col._offset, 0)
        self.assertEqual(len(col), 4)
        self.assertEqual(col.null_count(), 0)
        self.assertEqual(len(col._data), 4)
        self.assertEqual(list(col), list(range(4)))
        m = col[0 : len(col)]
        self.assertEqual(list(m), list(range(4)))
        with self.assertRaises(AttributeError):
            # AssertionError: can't append a finalized list
            col._append(None)

    def test_internals_full_nullable(self):
        col = self.ts.Column(dtype=dt.Int64(nullable=True))

        col = col.append([None, None, None])
        self.assertEqual(col.getdata(-1), dt.Int64(nullable=True).default)
        self.assertEqual(col.getmask(-1), True)

        col = col.append([3])
        self.assertEqual(col.getdata(-1), 3)
        self.assertEqual(col.getmask(-1), False)

        self.assertEqual(col.length(), 4)
        self.assertEqual(col.null_count(), 3)
        self.assertEqual(len(col), 4)

        self.assertEqual(col[0], None)
        self.assertEqual(col[3], 3)

        self.assertEqual(list(col), [None, None, None, 3])

        # # extend
        # col = col.append([4, 5])
        # self.assertEqual(list(col), [None, None, None, 3, 4, 5])

        # # len
        # self.assertEqual(len(col), 6)

    def test_internals_indexing(self):
        return self.base_test_internals_indexing()

    def test_boolean_column(self):
        return self.base_test_boolean_column()

    def test_infer(self):
        return self.base_test_internals_indexing()

    def test_map_where_filter(self):
        return self.base_test_map_where_filter()

    def test_reduce(self):
        return self.base_test_reduce()

    def test_sort_stuff(self):
        return self.base_test_sort_stuff()

    def test_operators(self):
        return self.base_test_operators()

    def test_na_handling(self):
        return self.base_test_na_handling()

    def test_agg_handling(self):
        return self.base_test_agg_handling()

    def test_in_nunique(self):
        return self.base_test_in_nunique()

    def test_math_ops(self):
        return self.base_test_math_ops()

    def test_describe(self):
        return self.base_test_describe()

    def test_batch_collate(self):
        return self.base_test_batch_collate()


if __name__ == "__main__":
    unittest.main()
