import unittest

import torcharrow as T


class TestNumericalColumnCpu(T.TestNumericalColumn):
    def setUp(self):
        self.ts = T.Scope({"device": "cpu"})

    def test_internals_empty(self):
        self.assertEqual(self.base_test_internals_empty(), "cpu")

    def test_internals_full(self):
        return self.base_test_internals_full()

    def test_internals_full_nullable(self):
        return self.base_test_internals_full_nullable()

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
