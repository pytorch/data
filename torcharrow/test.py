import functools
import statistics
import unittest

# datfrmes and columns
from torcharrow import (
    Column,
    _NumericalColumn,
    _ListColumn,
    _StringColumn,
    _MapColumn,
    _BooleanColumn,
    # _StructColumn,
)

# dataframe

from torcharrow import DataFrame

# schemas and dtypes
from torcharrow import (
    Field,
    Schema,
    int64,
    float64,
    Boolean,
    Int64,
    string,
    Float64,
    boolean,
    String,
    List_,
    Map,
    Struct,
    is_numerical,
)


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


class TestFlatColumn(unittest.TestCase):

    # build and iterate

    def test_flat(self):
        empty_i64_column = Column(int64)
        self.assertTrue(isinstance(empty_i64_column, _NumericalColumn))
        self.assertEqual(empty_i64_column._dtype, int64)
        self.assertEqual(len(empty_i64_column), 0)
        self.assertEqual(empty_i64_column._null_count, 0)
        self.assertEqual(len(empty_i64_column._data), 0)
        self.assertEqual(len(empty_i64_column._validity), 0)

        col = Column(int64)
        for i in range(4):
            col.append(i)
            self.assertEqual(col[i], i)
            self.assertEqual(col._validity[i], True)

        # self.assertEqual(col._offset, 0)
        self.assertEqual(len(col), 4)
        self.assertEqual(col._null_count, 0)
        self.assertEqual(len(col._data), 4)
        self.assertEqual(len(col._validity), 4)
        self.assertEqual(list(col), list(range(4)))
        self.assertEqual(list(col[0 : len(col)]), list(range(4)))
        with self.assertRaises(TypeError):
            # TypeError: an integer is required (got type NoneType)
            col.append(None)

        coln = Column(Int64(nullable=True))
        for i in [0, 1, 2]:
            coln.append(None)
            self.assertEqual(coln[i], None)
            self.assertEqual(coln._validity[i], False)
        for i in [3]:
            coln.append(i)
            self.assertEqual(coln[i], i)
            self.assertEqual(coln._validity[i], True)
        self.assertEqual(len(coln), 4)
        self.assertEqual(coln._null_count, 3)
        self.assertEqual(len(coln._data), 4)
        self.assertEqual(len(coln._validity), 4)
        self.assertEqual(list(coln), [None, None, None, 3])

        # extend
        coln.extend([4, 5])
        self.assertEqual(list(coln), [None, None, None, 3, 4, 5])

        # len
        self.assertEqual(len(coln), 6)

        # index
        self.assertEqual(coln[0], None)
        self.assertEqual(coln[-1], 5)
        self.assertEqual(list(coln[3 : len(coln)]), [3, 4, 5])

        # in
        coln2 = coln.fillna(99)
        self.assertEqual(list(coln[coln2.isin([4, 7])]), [4])


class TestNestedColumn(unittest.TestCase):

    # build and iterate

    def test_empty(self):
        empty_i64_column = Column(List_(int64))

        self.assertTrue(isinstance(empty_i64_column, _ListColumn))
        self.assertEqual(empty_i64_column._dtype, List_(int64))

        self.assertEqual(len(empty_i64_column), 0)
        self.assertEqual(empty_i64_column._null_count, 0)
        self.assertEqual(len(empty_i64_column._offsets), 1)
        self.assertEqual(empty_i64_column._offsets[0], 0)
        self.assertEqual(len(empty_i64_column._validity), 0)

    def test_nested_numerical_once(self):
        col = Column(List_(int64))
        for i in range(4):
            self.assertEqual(col._offsets[i], col._data.get_elements_size())
            col.append(list(range(i)))
            self.assertEqual(list(col[i]), list(range(i)))
            self.assertEqual(col._validity[i], True)
            self.assertEqual(col._offsets[i + 1], col._data.get_elements_size())

        verdict = [list(range(i)) for i in range(4)]

        for i, lst in zip(range(4), verdict):
            self.assertEqual(list(col[i]), lst)

        col.extend([[-1, -2, -3], [-4, -5]])
        # this is the last validity...
        # self.assertEqual(col[-1]._validity[-1], True)

    def test_nested_numerical_twice(self):
        col = Column(List_(List_(Int64(nullable=False), nullable=True), nullable=False))
        vals = [[[1, 2], None, [3, 4]], [[4], [5]]]
        col.append(vals[0])
        col.append(vals[1])
        self.assertEqual(vals, list(col))

    def test_nested_string_once(self):
        col = Column(List_(string))
        col.append([])
        col.append(["a"])
        col.append(["b", "c"])
        # s/elf.assertEqual(list([[],["a"],["b","c"]]),list(col))

    def test_nested_string_twice(self):
        col = Column(List_(List_(string)))
        col.append([])
        col.append([[]])
        col.append([["a"]])
        col.append([["b", "c"], ["d", "e", "f"]])
        self.assertEqual([[], [[]], [["a"]], [["b", "c"], ["d", "e", "f"]]], list(col))


class TestBooleanColumn(unittest.TestCase):
    def test_boolean(self):
        # setup
        col = Column(boolean)
        self.assertIsInstance(col, _BooleanColumn)

        col.extend([True, False, False])
        # tests
        self.assertEqual(list(col), [True, False, False])


class TestNumericalColumn(unittest.TestCase):
    def test_statistics(self):
        # setup
        coln = Column(Int64(nullable=True))
        coln.extend([None, None, None, 3, 4, 5])
        # tests
        self.assertEqual(coln.sum(), 3 + 4 + 5)
        self.assertEqual(coln.mean(), statistics.mean([3, 4, 5]))

    def test_operator(self):
        c = Column(int64)
        c.extend([1, 2, 3])

        # binary with constant
        d = c + 1
        self.assertEqual(list(d), [2, 3, 4])

        # binary with other column (same length, same type)
        e = c + c
        self.assertEqual(list(e), [2, 4, 6])

        # comparison
        f = e > c
        self.assertIsInstance(f, _BooleanColumn)
        self.assertEqual(list(f), [True, True, True])

        # filter
        g = e[f]
        self.assertEqual(list(g), list(e))

        h = Column(boolean)
        h.extend([True, False, True])

        i = e[h]
        self.assertEqual(list(i), [list(e)[0], list(e)[2]])

        # compose filter with condition
        self.assertEqual(list(e), [2, 4, 6])

        self.assertEqual(list(e[e > 5]), [e[-1]])

    def test_map(self):
        pass


class TestStringColumn(unittest.TestCase):
    def test_string(self):
        col = Column(string)
        self.assertIsInstance(col, _StringColumn)
        col.extend(["abc", "de", "", "f"])
        # tests
        self.assertEqual(list(col), ["abc", "de", "", "f"])


class TestMapColumn(unittest.TestCase):
    def test_string(self):
        col = Column(Map(string, int64))
        self.assertIsInstance(col, _MapColumn)

        col.append({"abc": 123})
        self.assertDictEqual(col[0], {"abc": 123})

        col.append({"de": 45, "fg": 67})
        self.assertDictEqual(col[1], {"de": 45, "fg": 67})


class TestStructColumn(unittest.TestCase):
    def test_string(self):
        col = Column(Struct([Field("a", string), Field("b", int64)]))

        self.assertIsInstance(col, _StructColumn)

        col.append(("a", 10))
        self.assertEqual(col[0], ("a", 10))

        self.assertIsInstance(col["a"], _StringColumn)
        self.assertIsInstance(col["b"], _NumericalColumn)

        self.assertEqual(col["a"][0], "a")


class TestFunc(unittest.TestCase):
    def test_functool(self):
        col = Column([1, 2, 3, 4])
        odd = col.filter(lambda x: x % 2 == 0)
        self.assertEqual(list(odd), list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4])))

        succ = col.map(lambda x: x * 2)
        self.assertEqual(list(succ), list(map(lambda x: x * 2, [1, 2, 3, 4])))

        res = col.reduce(lambda res, elem: res * elem)
        self.assertEqual(
            res, functools.reduce(lambda res, elem: res * elem, [1, 2, 3, 4])
        )

        col = Column(List_(string))
        col.append(["hello", "world"])
        res = col.map(lambda xs: " ".join(xs), string)
        self.assertEqual(res[0], "hello world")

        col = Column(string)
        col.append("hello")
        col.append("world")

        res = col.flatmap(lambda x: [x, x])
        self.assertEqual(res[0], "hello")
        self.assertEqual(res[1], "hello")


class TestFrom(unittest.TestCase):
    def test_from(self):

        # not enough info
        with self.assertRaises(ValueError):
            Column([])

        # int
        c = Column([1])
        self.assertEqual(c._dtype, int64)
        self.assertEqual(list(c), [1])

        # bool
        c = Column([True, None])
        self.assertEqual(c._dtype, Boolean(nullable=True))
        self.assertEqual(list(c), [True, None])

        # inconsistent info
        with self.assertRaises(ValueError):
            Column([True, 1])

        # string
        c = Column(["a", "b"])
        self.assertEqual(c._dtype, string)
        self.assertEqual(list(c), ["a", "b"])

        # float
        c = Column([1, 2.0])
        self.assertEqual(c._dtype, float64)
        self.assertEqual(list(c), [1.0, 2.0])

        # list
        # nested types not yet impemented
        with self.assertRaises(NotImplementedError):
            c = Column([[1], None, [2.0]])
            self.assertEqual(c.dtype, List_(Float64(nullable=True)))
            self.assertEqual(list(c), [[1], None, [2.0]])


class TestDataframe(unittest.TestCase):
    def test_dataframe(self):
        df = DataFrame(Schema([Field("a", string), Field("b", int64)]))
        self.assertIsInstance(df, _StructColumn)

        # row wise operation..
        df.append(("a", 10))
        self.assertEqual(df[0], ("a", 10))

        self.assertIsInstance(df["a"], _StringColumn)
        self.assertIsInstance(df["b"], _NumericalColumn)

        self.assertEqual(df["a"][0], "a")

        gf = DataFrame({"a": [1, 2, 3], "b": ["abc", "de", "f"]})
        self.assertIsInstance(gf["b"], _StringColumn)
        self.assertIsInstance(gf["a"], _NumericalColumn)

        # All operations on structs are promoted...
        hf = gf.filter(lambda x: x[0] > 1)
        self.assertEqual(hf["a"][0], 2)

    def test_dataframe_setter_getter(self):
        # empty datframe

        # column wise operation
        df = DataFrame()
        self.assertEqual(len(df), 0)
        self.assertEqual(df.dtype, Schema([]))
        self.assertEqual(df.columns(), {})

        # Currently setter/getter not supported
        # df.a = Column([111,222,333])
        # self.assertEqual(len(df),3)
        # self.assertEqual(df.a,Column([111,222,333]))

        # But setitem/getitem is supported
        df["a"] = Column([111, 222, 333])
        df["b"] = Column(["a", "b", "c"])

        self.assertEqual(len(df), 3)
        self.assertEqual(df["b"], Column(["a", "b", "c"]))

    def test_dataframe_drop_etc(self):
        df = DataFrame()
        df["a"] = Column([111, 222, 333])
        df["b"] = Column(["a", "b", "c"])
        self.assertEqual([f.name for f in df.dtype.fields], ["a", "b"])
        gf = df.drop(["a"])
        self.assertEqual([f.name for f in gf.dtype.fields], ["b"])


if __name__ == "__main__":
    unittest.main()
