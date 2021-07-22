import unittest

import torcharrow.dtypes as dt
from torcharrow import Scope, IStringColumn

from .test_string_column import TestStringColumn

from torcharrow.velox_rt.functional import functional


class TestStringColumnCpu(TestStringColumn):
    def setUp(self):
        self.ts = Scope({"device": "cpu"})

    def test_empty(self):
        self.base_test_empty()

    def test_append_offsets(self):
        self.base_test_append_offsets()

    def test_string_split_methods(self):
        self.base_test_string_split_methods()

    def test_string_lifted_methods(self):
        self.base_test_string_lifted_methods()

    def test_regular_expressions(self):
        self.base_test_regular_expressions()

    def test_functional(self):
        str_col = self.ts.Column(["", "abc", "XYZ", "123", "xyz123", None])

        self.assertEqual(
            list(functional.torcharrow_isalpha(str_col)),
            [False, True, True, False, False, None]
        )

        self.assertEqual(
            list(functional.upper(str_col)),
            ["", "ABC", "XYZ", "123", "XYZ123", None]
        )


if __name__ == "__main__":
    unittest.main()
