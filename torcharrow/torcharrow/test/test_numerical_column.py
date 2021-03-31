import functools
import statistics
import unittest
from decimal import ExtendedContext
from math import ceil, floor
from os import initgroups
from typing import AsyncGenerator

from torcharrow import (
    Boolean,
    BooleanColumn,
    Column,
    Float64,
    Int64,
    NumericalColumn,
    String,
    boolean,
    float64,
    int64,
    is_numerical,
    string,
)

# run python3 -m unittest outside this directory to run all tests


class TestNumericalColumn(unittest.TestCase):
    def test_imternals_empty(self):
        empty_i64_column = Column(int64)

        # testing internals...
        self.assertTrue(isinstance(empty_i64_column, NumericalColumn))
        self.assertEqual(empty_i64_column._dtype, int64)
        self.assertEqual(empty_i64_column._length, 0)
        self.assertEqual(empty_i64_column._null_count, 0)
        self.assertEqual(len(empty_i64_column._data), 0)
        self.assertEqual(len(empty_i64_column._validity), 0)

    def test_internals_full(self):
        col = Column(int64)
        for i in range(4):
            col.append(i)
            self.assertEqual(col[i], i)
            self.assertEqual(col._validity[i], True)

        self.assertEqual(col._offset, 0)
        self.assertEqual(col._length, 4)
        self.assertEqual(col._null_count, 0)
        self.assertEqual(len(col._data), 4)
        self.assertEqual(len(col._validity), 4)
        self.assertEqual(list(col), list(range(4)))
        m = col[0: len(col)]
        self.assertEqual(list(col[0: len(col)]), list(range(4)))
        with self.assertRaises(TypeError):
            # TypeError: an integer is required (got type NoneType)
            col.append(None)

    def test_internals_full_nullable(self):
        col = Column(Int64(nullable=True))
        for i in [0, 1, 2]:
            col.append(None)
            # note this adds the default value to column,
            self.assertEqual(col._data[-1], 0)
            # but all public APIs report this back as None
            self.assertEqual(col[i], None)
            self.assertEqual(col._validity[i], False)
            self.assertEqual(col._null_count, i + 1)
        for i in [3]:
            col.append(i)
            self.assertEqual(col[i], i)
            self.assertEqual(col._validity[i], True)

        self.assertEqual(col._length, 4)
        self.assertEqual(col._null_count, 3)
        self.assertEqual(len(col._data), 4)
        self.assertEqual(len(col._validity), 4)
        self.assertEqual(col._offset, 0)

        self.assertEqual(col[0], None)
        self.assertEqual(col[3], 3)

        self.assertEqual(list(col), [None, None, None, 3])

        # extend
        col.extend([4, 5])
        self.assertEqual(list(col), [None, None, None, 3, 4, 5])

        # len
        self.assertEqual(len(col), 6)

    def test_internals_indexing(self):
        col = Column(Int64(nullable=True))
        col.extend([None] * 3)
        col.extend([3, 4, 5])

        # index
        self.assertEqual(col[0], None)
        self.assertEqual(col[-1], 5)

        # slice

        # continuous slice creates a view
        c = col[3: len(col)]
        self.assertEqual(len(col), 6)
        self.assertEqual(len(c), 3)
        self.assertEqual(id(col._data), id(c._data))

        # non continuous slice creates new data
        d = col[::2]
        self.assertEqual(len(col), 6)
        self.assertEqual(len(d), 3)
        self.assertNotEqual(id(col._data), id(d._data))

        # TorchArrow integer slice has Python not Panda semantics
        # that is last index is exclusive.
        e = col[: len(col) - 1]
        self.assertEqual(len(e), len(col) - 1)

        # indexing via lists
        f = col[[0, 1, 2]]
        self.assertEqual(list(col[[0, 1, 2]]), list(col[:3]))

        # indexing via lists

        self.assertEqual(list(col.head(2)), [None, None])
        self.assertEqual(list(col.tail(2)), [4, 5])

    def test_boolean_column(self):

        col = Column(boolean)
        self.assertIsInstance(col, BooleanColumn)

        col.extend([True, False, False])
        # tests
        self.assertEqual(list(col), [True, False, False])

        # numerics can be converted to booleans...
        col.append(1)
        self.assertEqual(list(col), [True, False, False, True])

    def test_metastuff(self):
        col = Column(Int64(nullable=True))
        col.extend([None] * 3)
        col.extend([3, 4, 5])

        self.assertEqual(col.dtype, Int64(nullable=True))
        self.assertEqual(col.ndim, 1)
        self.assertEqual(col.size, len(col))
        self.assertTrue(col.isnullable)
        self.assertTrue(col.is_appendable)

        self.assertEqual(list(col), list(col.copy(deep=False)))
        self.assertEqual(list(col), list(col.copy(deep=True)))

    def test_infer(self):

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

        # float
        c = Column([1, 2.0])
        self.assertEqual(c._dtype, float64)
        self.assertEqual(list(c), [1.0, 2.0])

        # string:: to do in a different tests
        # ...

        # list:: to do in a different tests
        # ...

        # map:: to do in a different tests
        # ...

        # dataframe:: to do in a different tests
        # ...

    def test_map_where_filter(self):
        col = Column(Int64(nullable=True))
        col.extend([None] * 3)
        col.extend([3, 4, 5])

        # keep None
        self.assertEqual(list(col.map({3: 33})), [
                         None, None, None, 33, None, None])

        # maps None
        self.assertEqual(
            list(col.map({None: 1, 3: 33}, na_action="ignore")),
            [1, 1, 1, 33, None, None],
        )

        # keep None
        self.assertEqual(
            list(col.map({None: 1, 3: 33}, na_action=None)),
            [None, None, None, 33, None, None],
        )

        # maps as function
        self.assertEqual(
            list(
                col.map(
                    lambda x: 1 if x is None else 33 if x == 3 else x,
                    na_action="ignore",
                )
            ),
            [1, 1, 1, 33, 4, 5],
        )

        # # where
        # TODO this should be moved to conditional (or so) (with two or three args)
        # self.assertEqual(
        #     list(col.where([True, False] * 3, [i * i for i in range(6)])),
        #     [None, 1, None, 9, 4, 25],
        # )

        # filter
        self.assertEqual(list(col.filter([True, False] * 3)), [None, None, 4])

    def test_sort_stuff(self):
        col = Column([1, 2, 3])
        self.assertEqual(list(col.sort_values()), [1, 2, 3])
        self.assertEqual(list(col.sort_values(ascending=False)), [3, 2, 1])
        self.assertEqual(
            list(Column([None, 1, 5, 2]).sort_values()), [1, 2, 5, None])
        self.assertEqual(
            list(Column([None, 1, 5, 2]).sort_values(na_position="first")),
            [None, 1, 2, 5],
        )

        self.assertEqual(
            list(Column([None, 1, 5, 2]).nlargest(n=2, keep="first")), [5, 2]
        )
        self.assertEqual(
            list(Column([None, 1, 5, 2]).nsmallest(n=2, keep="last")), [1, 2]
        )
        self.assertEqual(
            list(Column([None, 1, 5, 2]).reverse()), [2, 5, 1, None])

    def test_operators(self):
        # without None
        c = Column([0, 1, 3])
        d = Column([5, 5, 6])
        e = Column([1.0, 1, 7])

        # ==, !=

        self.assertEqual(list(c == c), [True] * 3)
        self.assertEqual(list(c == d), [False] * 3)

        # NOTE: Yoo cannot compare Columns with assertEqual,
        #       since torcharrow overrode __eq__
        #       this always compare with list(), etc
        #       or write (a==b).all()

        self.assertEqual(list(c == 1), [False, True, False])
        self.assertTrue((c == 1) == Column([False, True, False]).all())

        # <, <=, >=, >

        self.assertEqual(list(c <= 2), [True, True, False])
        self.assertEqual(list(c < d), [True, True, True])
        self.assertEqual(list(c >= d), [False, False, False])
        self.assertEqual(list(c > 2), [False, False, True])

        # +,-,*,/,//,**

        self.assertEqual(list(-c), [0, -1, -3])
        self.assertEqual(list(+-c), [0, -1, -3])

        self.assertEqual(list(c + 1), [1, 2, 4])
        self.assertEqual(list(c.add(1)), [1, 2, 4])

        self.assertEqual(list(1 + c), [1, 2, 4])
        self.assertEqual(list(c.radd(1)), [1, 2, 4])

        self.assertEqual(list(c + d), [5, 6, 9])
        self.assertEqual(list(c.add(d)), [5, 6, 9])

        self.assertEqual(list(c + 1), [1, 2, 4])
        self.assertEqual(list(1 + c), [1, 2, 4])
        self.assertEqual(list(c + d), [5, 6, 9])

        self.assertEqual(list(c - 1), [-1, 0, 2])
        self.assertEqual(list(1 - c), [1, 0, -2])
        self.assertEqual(list(d - c), [5, 4, 3])

        self.assertEqual(list(c * 2), [0, 2, 6])
        self.assertEqual(list(2 * c), [0, 2, 6])
        self.assertEqual(list(c * d), [0, 5, 18])

        self.assertEqual(list(c * 2), [0, 2, 6])
        self.assertEqual(list(2 * c), [0, 2, 6])
        self.assertEqual(list(c * d), [0, 5, 18])

        self.assertEqual(list(c / 2), [0.0, 0.5, 1.5])

        with self.assertRaises(ZeroDivisionError):
            self.assertEqual(list(2 / c), [0.0, 0.5, 1.5])
        self.assertEqual(list(c / d), [0.0, 0.2, 0.5])

        self.assertEqual(list(d // 2), [2, 2, 3])
        self.assertEqual(list(2 // d), [0, 0, 0])
        self.assertEqual(list(c // d), [0, 0, 0])
        self.assertEqual(list(e // d), [0.0, 0.0, 1.0])
        # THIS ASSERTION SHOULD NOT HAPPEN, FIX derive_dtype
        # TypeError: integer argument expected, got float
        with self.assertRaises(TypeError):
            self.assertEqual(list(d // e), [0.0, 0.0, 1.0])

        self.assertEqual(list(c ** 2), [0, 1, 9])
        self.assertEqual(list(2 ** c), [1, 2, 8])
        self.assertEqual(list(c ** d), [0, 1, 729])

        # null handling

        c = Column([0, 1, 3, None])
        self.assertEqual(list(c.add(1)), [1, 2, 4, None])

        self.assertEqual(list(c.add(1, fill_value=17)), [1, 2, 4, 18])
        self.assertEqual(list(c.radd(1, fill_value=-1)), [1, 2, 4, 0])
        f = Column([None, 1, 3, None])
        self.assertEqual(list(c.radd(f, fill_value=100)), [100, 2, 6, 200])

        # &, |, ~
        g = Column([True, False, True, False])
        h = Column([False, False, True, True])
        self.assertEqual(list(g & h), [False, False, True, False])
        self.assertEqual(list(g | h), [True, False, True, True])
        self.assertEqual(list(~g), [False, True, False, True])

    def test_na_handling(self):
        c = Column([None, 2, 17.0])

        self.assertEqual(list(c.fillna(99.0)), [99.0, 2, 17.0])
        self.assertEqual(list(c.dropna()), [2, 17.0])

        c.append(2)
        self.assertEqual(list(c.drop_duplicates()), [None, 2, 17.0])

    def test_agg_handling(self):
        import functools
        import operator

        c = [1, 4, 2, 7, 9, 0]
        C = Column(c + [None])

        self.assertEqual(C.min(), min(c))
        self.assertEqual(C.max(), max(c))
        self.assertEqual(C.sum(), sum(c))
        self.assertEqual(C.prod(), functools.reduce(operator.mul, c, 1))
        self.assertEqual(C.mode(), statistics.mode(c))
        self.assertEqual(C.std(), statistics.stdev(c))
        self.assertEqual(C.mean(), statistics.mean(c))
        self.assertEqual(C.median(), statistics.median(c))

        self.assertEqual(
            list(C.cummin()), [min(c[:i])
                               for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C.cummax()), [max(c[:i])
                               for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C.cumsum()), [sum(c[:i])
                               for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C.cumprod()),
            [functools.reduce(operator.mul, c[:i], 1)
             for i in range(1, len(c) + 1)]
            + [None],
        )
        self.assertEqual((C % 2 == 0)[:-1].all(), all(i % 2 == 0 for i in c))
        self.assertEqual((C % 2 == 0)[:-1].any(), any(i % 2 == 0 for i in c))

    def test_in_nunique(self):
        c = [1, 4, 2, 7]
        C = Column(c + [None])
        self.assertEqual(list(C.isin([1, 2, 3])), [
                         True, False, True, False, False])
        C.extend(c)
        d = set(c)
        d.add(None)
        self.assertEqual(C.nunique(), len(set(C) - {None}))
        self.assertEqual(C.nunique(dropna=False), len(set(C)))

        self.assertEqual(C.is_unique(), False)
        self.assertEqual(Column([1, 2, 3]).is_unique(), True)

        self.assertEqual(Column([1, 2, 3]).is_monotonic_increasing(), True)
        self.assertEqual(Column(int64).is_monotonic_decreasing(), True)

    def test_math_ops(self):
        c = [1.0, 4.2, 2, 7, -9, -2.5]
        C = Column(c + [None])

        self.assertEqual(list(C.abs()), [abs(i) for i in c] + [None])
        self.assertEqual(list(C.ceil()), [ceil(i) for i in c] + [None])
        self.assertEqual(list(C.floor()), [floor(i) for i in c] + [None])

        self.assertEqual(list(C.round()), [round(i) for i in c] + [None])
        self.assertEqual(list(C.round(2)), [round(i, 2) for i in c] + [None])
        self.assertEqual(list(C.hash_values()), [hash(i) for i in c] + [None])

    def test_describe(self):
        # requires 'implicitly' torcharrow.dataframe import DataFrame
        c = Column([1, 2, 3])
        self.assertEqual(
            list(c.describe()),
            [
                ("count", 3.0),
                ("mean", 2.0),
                ("std", 1.0),
                ("min", 1.0),
                ("25%", 1.5),
                ("50%", 2.0),
                ("75%", 2.5),
                ("max", 3.0),
            ],
        )

    def test_mutability(self):
        x = Column([0, 1, 2, 3])
        y = x[:3]
        x.append(44)
        with self.assertRaises(AttributeError):
            # AttributeError: column is not appendable
            y.append(33)
        y = y.copy()
        y.append(33)
        self.assertEqual(list(y), list(Column([0, 1, 2, 33])))
        self.assertEqual(list(x), list(Column([0, 1, 2, 3, 44])))


if __name__ == "__main__":

    unittest.main()
