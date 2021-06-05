import unittest

import torcharrow.dtypes as dt
from torcharrow import Scope, IStringColumn

from .test_string_column import TestStringColumn


class TestStringColumnStd(TestStringColumn):
    def setUp(self):
        self.ts = Scope({"device": "std"})

    def test_empty(self):
        empty = self.ts.Column(dt.string)
        self.assertEqual(len(empty._data), 0)
        self.assertEqual(len(empty._mask), 0)

        self.base_test_empty()

    def test_append_offsets(self):
        self.base_test_append_offsets()

    def test_string_split_methods(self):
        self.base_test_string_split_methods()

    def test_string_lifted_methods(self):
        self.base_test_string_lifted_methods()

    def test_regular_expressions(self):
        self.base_test_regular_expressions()


if __name__ == "__main__":
    unittest.main()
