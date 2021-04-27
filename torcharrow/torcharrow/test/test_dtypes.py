import unittest

# schemas and dtypes
import typing
from torcharrow import (
    Boolean,
    Field,
    Float64,
    Int64,
    List_,
    Tuple_,
    Map,
    Schema,
    String,
    Struct,
    boolean,
    float64,
    int64,
    is_numerical,
    string,
    from_type_hint,
)

# run python3 -m unittest outside this directory to run all tests


class TestTypes(unittest.TestCase):
    def test_numericals(self):
        # plain type
        self.assertEqual(str(int64), "int64")
        self.assertEqual(int64.size, 8)
        self.assertEqual(int64.name, "int64")
        self.assertEqual(int64.typecode, "l")
        self.assertEqual(int64.arraycode, "l")
        self.assertTrue(is_numerical(int64))

    def test_string(self):
        # plain type
        self.assertEqual(str(string), "string")
        self.assertEqual(string.typecode, "u")
        self.assertEqual(string.nullable, False)
        self.assertEqual(String(nullable=True).nullable, True)
        self.assertEqual(string.size, -1)

    def test_list(self):
        self.assertEqual(
            str(List_(Int64(nullable=True))), "List_(Int64(nullable=True))"
        )
        self.assertEqual(
            str(List_(Int64(nullable=True)).item_dtype), "Int64(nullable=True)"
        )
        self.assertEqual(List_(Int64(nullable=True)).typecode, "+l")
        self.assertEqual(List_(int).size, -1)

    def test_map(self):
        self.assertEqual(str(Map(int64, string)), "Map(int64, string)")
        self.assertEqual(Map(int64, string).typecode, "+m")

    def test_struct(self):
        self.assertEqual(
            str(Struct([Field("a", int64), Field("b", string)])),
            "Struct([Field('a', int64), Field('b', string)])",
        )
        self.assertEqual(Struct([Field("a", int64), Field("b", string)]).typecode, "+s")

    def test_annotations(self):
        self.assertEqual(from_type_hint(typing.Optional[int]), Int64(nullable=True))
        self.assertEqual(from_type_hint(typing.List[str]), List_(string))
        self.assertEqual(
            from_type_hint(typing.Dict[str, int64.with_null()]),
            Map(string, Int64(nullable=True)),
        )
        self.assertEqual(
            from_type_hint(typing.Tuple[int, float]), Tuple_([int64, float64])
        )

        class FooTuple(typing.NamedTuple):
            a: int
            b: float

        self.assertEqual(
            from_type_hint(FooTuple), Struct([Field("a", int64), Field("b", float64)])
        )


if __name__ == "__main__":
    unittest.main()
