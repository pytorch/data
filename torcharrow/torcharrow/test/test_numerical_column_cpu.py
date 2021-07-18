import unittest

import torcharrow.dtypes as dt
from torcharrow import INumericalColumn
from torcharrow import Scope

from .test_numerical_column import TestNumericalColumn


class TestNumericalColumnCpu(TestNumericalColumn):
    def setUp(self):
        self.ts = Scope({"device": "cpu"})

    def test_internal_empty(self):
        c = self.base_test_empty()
        self.assertEqual(c.device, "cpu")
        # internals...self.assertEqual(len(c._mask), 0)

    def test_internals_full(self):
        c = self.base_test_full()
        #  internals... self.assertEqual(len(c._data), len(c))
        #  internals... self.assertEqual(len(c._mask), len(c))

    def test_internals_full_nullable(self):
        return self.base_test_full_nullable()

    def test_is_immutable(self):
        return self.base_test_is_immutable()

    def test_internals_indexing(self):
        return self.base_test_indexing()

    def test_boolean_column(self):
        return self.base_test_boolean_column()

    def test_infer(self):
        return self.base_test_infer()

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
