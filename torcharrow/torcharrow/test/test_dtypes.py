import unittest

from torcharrow.dtypes import (
    Field,
    Int64,
    List,
    Map,
    String,
    Struct,
    int64,
    is_numerical,
    string,
)


class TestTypes(unittest.TestCase):
    def test_int64(self):
        self.assertEqual(str(int64), "int64")
        self.assertEqual(int64.name, "int64")
        self.assertEqual(int64.typecode, "l")
        self.assertEqual(int64.arraycode, "l")
        self.assertTrue(is_numerical(int64))

    def test_string(self):
        self.assertEqual(str(string), "string")
        self.assertEqual(string.typecode, "u")
        self.assertEqual(string.nullable, False)
        self.assertEqual(String(nullable=True).nullable, True)

    def test_list(self):
        self.assertEqual(str(List(Int64(nullable=True))), "List(Int64(nullable=True))")
        self.assertEqual(
            str(List(Int64(nullable=True)).item_dtype), "Int64(nullable=True)"
        )
        self.assertEqual(List(Int64(nullable=True)).typecode, "+l")

    def test_map(self):
        self.assertEqual(str(Map(int64, string)), "Map(int64, string)")
        self.assertEqual(Map(int64, string).typecode, "+m")

    def test_struct(self):
        self.assertEqual(
            str(Struct([Field("a", int64), Field("b", string)])),
            "Struct([Field('a', int64), Field('b', string)])",
        )
        self.assertEqual(Struct([Field("a", int64), Field("b", string)]).typecode, "+s")


if __name__ == "__main__":
    unittest.main()
