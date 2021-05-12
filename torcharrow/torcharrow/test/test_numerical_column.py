import functools
import statistics
import unittest
from math import ceil, floor

import torcharrow.dtypes as dt
from torcharrow import IColumn, INumericalColumn


class TestNumericalColumn(unittest.TestCase):
    # same base tests will be called by
    # - TestNumericalColumnCpu
    # - TestNumericalColumnTest
    # but with different setups
    # def setUp(self):
    #     self.ts = Scope({"device": "cpu"})

    def base_test_internals_empty(self):
        empty_i64_column = self.ts.Column(dtype=dt.int64)

        # testing internals...
        self.assertTrue(isinstance(empty_i64_column, INumericalColumn))
        self.assertEqual(empty_i64_column.dtype, dt.int64)
        self.assertEqual(len(empty_i64_column), 0)
        self.assertEqual(empty_i64_column.null_count(), 0)
        self.assertEqual(len(empty_i64_column), 0)
        self.assertEqual(len(empty_i64_column._mask), 0)
        return empty_i64_column.to

    def base_test_internals_full(self):
        col = self.ts.Column([i for i in range(4)], dtype=dt.int64)

        # self.assertEqual(col._offset, 0)
        self.assertEqual(len(col), 4)
        self.assertEqual(col.null_count(), 0)
        self.assertEqual(len(col._data), 4)
        self.assertEqual(len(col._mask), 4)
        self.assertEqual(list(col), list(range(4)))
        m = col[0 : len(col)]
        self.assertEqual(list(m), list(range(4)))
        with self.assertRaises(AttributeError):
            # AssertionError: can't append a finalized list
            col._append(None)

    def base_test_internals_full_nullable(self):
        col = self.ts.Column(dtype=dt.Int64(nullable=True))

        col = col.append([None, None, None])
        self.assertEqual(col._data[-1], dt.Int64(nullable=True).default)
        self.assertEqual(col._mask[-1], True)

        col = col.append([3])
        self.assertEqual(col._data[-1], 3)
        self.assertEqual(col._mask[-1], False)

        self.assertEqual(col.length(), 4)
        self.assertEqual(col.null_count(), 3)
        self.assertEqual(len(col._data), 4)
        self.assertEqual(len(col._mask), 4)

        self.assertEqual(col[0], None)
        self.assertEqual(col[3], 3)

        self.assertEqual(list(col), [None, None, None, 3])

        # # extend
        # col = col.append([4, 5])
        # self.assertEqual(list(col), [None, None, None, 3, 4, 5])

        # # len
        # self.assertEqual(len(col), 6)

    def base_test_internals_indexing(self):
        col = self.ts.Column([None] * 3 + [3, 4, 5], dtype=dt.Int64(nullable=True))

        # index
        self.assertEqual(col[0], None)
        self.assertEqual(col[-1], 5)

        # slice

        # continuous slice
        c = col[3 : len(col)]
        self.assertEqual(len(col), 6)
        self.assertEqual(len(c), 3)

        # non continuous slice
        d = col[::2]
        self.assertEqual(len(col), 6)
        self.assertEqual(len(d), 3)

        # slice has Python not Panda semantics
        e = col[: len(col) - 1]
        self.assertEqual(len(e), len(col) - 1)

        # indexing via lists
        f = col[[0, 1, 2]]
        self.assertEqual(list(f), list(col[:3]))

        # head/tail are special slices
        self.assertEqual(list(col.head(2)), [None, None])
        self.assertEqual(list(col.tail(2)), [4, 5])

    def base_test_boolean_column(self):

        col = self.ts.Column(dt.boolean)
        self.assertIsInstance(col, INumericalColumn)

        col = col.append([True, False, False])
        self.assertEqual(list(col), [True, False, False])

        # numerics can be converted to booleans...
        col = col.append([1])
        self.assertEqual(list(col), [True, False, False, True])

    def base_test_infer(self):

        # not enough info
        with self.assertRaises(ValueError):
            self.ts.Column([])

        # int
        c = self.ts.Column([1])
        self.assertEqual(c._dtype, dt.int64)
        self.assertEqual(list(c), [1])

        # bool
        c = self.ts.Column([True, None])
        self.assertEqual(c._dtype, dt.Boolean(nullable=True))
        self.assertEqual(list(c), [True, None])

        # inconsistent info
        with self.assertRaises(ValueError):
            self.ts.Column([True, 1])

        # float
        c = self.ts.Column([1, 2.0])
        self.assertEqual(c._dtype, dt.float64)
        self.assertEqual(list(c), [1.0, 2.0])

        # dt.string:: to do in a different tests
        # ...

        # list:: to do in a different tests
        # ...

        # map:: to do in a different tests
        # ...

        # dataframe:: to do in a different tests
        # ...

    def base_test_map_where_filter(self):
        col = self.ts.Column([None] * 3 + [3, 4, 5], dtype=dt.Int64(nullable=True))

        # Values that are not found in the dict are converted to None
        self.assertEqual(list(col.map({3: 33})), [None, None, None, 33, None, None])

        # maps None
        self.assertEqual(
            list(col.map({None: 1, 3: 33})),
            [1, 1, 1, 33, None, None],
        )

        # propagates None
        self.assertEqual(
            list(col.map({None: 1, 3: 33}, na_action="ignore")),
            [None, None, None, 33, None, None],
        )

        # maps as function
        self.assertEqual(
            list(
                col.map(
                    lambda x: 1 if x is None else 33 if x == 3 else x,
                )
            ),
            [1, 1, 1, 33, 4, 5],
        )

        self.assertEqual(
            list((col > 3).ite([0] * 6, [99] * 6)), [None, None, None, 99, 0, 0]
        )

        # filter
        self.assertEqual(list(col.filter([True, False] * 3)), [None, None, 4])

    @staticmethod
    def _accumulate(col, val):
        if len(col) == 0:
            col._append(val)
        else:
            col._append(col[-1] + val)
        return col

    @staticmethod
    def _finalize(col):
        return col._finalize()

    def base_test_reduce(self):
        c = self.ts.Column([1, 2, 3])
        d = c.reduce(
            fun=TestNumericalColumn._accumulate,
            initializer=self.ts._EmptyColumn(dt.int64),
            finalizer=TestNumericalColumn._finalize,
        )
        self.assertEqual(list(self.ts.Column(d, dtype=c.dtype)), [1, 3, 6])

    def base_test_sort_stuff(self):
        col = self.ts.Column([2, 1, 3])

        self.assertEqual(list(col.sort()), [1, 2, 3])
        self.assertEqual(list(col.sort(ascending=False)), [3, 2, 1])
        self.assertEqual(list(self.ts.Column([None, 1, 5, 2]).sort()), [1, 2, 5, None])
        self.assertEqual(
            list(self.ts.Column([None, 1, 5, 2]).sort(na_position="first")),
            [None, 1, 2, 5],
        )
        self.assertEqual(
            list(self.ts.Column([None, 1, 5, 2]).sort(na_position="last")),
            [1, 2, 5, None],
        )

        self.assertEqual(
            list(self.ts.Column([None, 1, 5, 2]).sort(na_position="last")),
            [1, 2, 5, None],
        )

        self.assertEqual(
            list(self.ts.Column([None, 1, 5, 2]).nlargest(n=2, keep="first")), [5, 2]
        )
        self.assertEqual(
            list(self.ts.Column([None, 1, 5, 2]).nsmallest(n=2, keep="last")), [1, 2]
        )
        self.assertEqual(
            list(self.ts.Column([None, 1, 5, 2]).reverse()), [2, 5, 1, None]
        )

    def base_test_operators(self):
        # without None
        c = self.ts.Column([0, 1, 3])
        d = self.ts.Column([5, 5, 6])
        e = self.ts.Column([1.0, 1, 7])

        # ==, !=

        self.assertEqual(list(c == c), [True] * 3)
        self.assertEqual(list(c == d), [False] * 3)

        # NOTE: Yoo cannot compare Columns with assertEqual,
        #       since torcharrow overrode __eq__
        #       this always compare with list(), etc
        #       or write (a==b).all()

        self.assertEqual(list(c == 1), [False, True, False])
        self.assertTrue((c == 1) == self.ts.Column([False, True, False]).all())
        self.assertTrue((1 == c) == self.ts.Column([False, True, False]).all())

        # <, <=, >=, >

        self.assertEqual(list(c <= 2), [True, True, False])
        self.assertEqual(list(c < d), [True, True, True])
        self.assertEqual(list(c >= d), [False, False, False])
        self.assertEqual(list(c > 2), [False, False, True])

        # # +,-,*,/,//,**

        self.assertEqual(list(-c), [0, -1, -3])
        self.assertEqual(list(+-c), [0, -1, -3])

        self.assertEqual(list(c + 1), [1, 2, 4])
        # self.assertEqual(list(c.add(1)), [1, 2, 4])

        self.assertEqual(list(1 + c), [1, 2, 4])
        # self.assertEqual(list(c.radd(1)), [1, 2, 4])

        self.assertEqual(list(c + d), [5, 6, 9])
        # self.assertEqual(list(c.add(d)), [5, 6, 9])

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

        self.assertEqual(
            [round(i, 2) if i is not None else None for i in list(2 / c)],
            [round(i, 2) if i is not None else None for i in [None, 2.0, 0.66666667]],
        )

        self.assertEqual(list(c / d), [0.0, 0.2, 0.5])

        self.assertEqual(list(d // 2), [2, 2, 3])
        self.assertEqual(list(2 // d), [0, 0, 0])
        self.assertEqual(list(c // d), [0, 0, 0])
        self.assertEqual(list(e // d), [0.0, 0.0, 1.0])

        self.assertEqual(list(d // e), [5.0, 5.0, 0.0])

        self.assertEqual(list(c ** 2), [0, 1, 9])
        self.assertEqual(list(2 ** c), [1, 2, 8])
        self.assertEqual(list(c ** d), [0, 1, 729])

        # TODO: Decide ...null handling.., bring back or ignore

        # c = self.ts.Column([0, 1, 3, None])
        # self.assertEqual(list(c.add(1)), [1, 2, 4, None])

        # self.assertEqual(list(c.add(1, fill_value=17)), [1, 2, 4, 18])
        # self.assertEqual(list(c.radd(1, fill_value=-1)), [1, 2, 4, 0])
        # f = self.ts.Column([None, 1, 3, None])
        # self.assertEqual(list(c.radd(f, fill_value=100)), [100, 2, 6, 200])

        # &, |, ~
        g = self.ts.Column([True, False, True, False])
        h = self.ts.Column([False, False, True, True])
        self.assertEqual(list(g & h), [False, False, True, False])
        self.assertEqual(list(g | h), [True, False, True, True])
        self.assertEqual(list(~g), [False, True, False, True])

    def base_test_na_handling(self):
        c = self.ts.Column([None, 2, 17.0])

        self.assertEqual(list(c.fillna(99.0)), [99.0, 2, 17.0])
        self.assertEqual(list(c.dropna()), [2.0, 17.0])

        c = c.append([2])
        self.assertEqual(set(c.drop_duplicates()), {None, 2, 17.0})

    def base_test_agg_handling(self):
        import functools
        import operator

        c = [1, 4, 2, 7, 9, 0]
        D = self.ts.Column(c)
        C = self.ts.Column(c + [None])

        self.assertEqual(C.min(), min(c))
        self.assertEqual(C.max(), max(c))
        self.assertEqual(C.sum(), sum(c))
        self.assertEqual(C.prod(), functools.reduce(operator.mul, c, 1))

        self.assertEqual(D.std(), (statistics.stdev((float(i) for i in c))))

        self.assertEqual(C.std(), (statistics.stdev((float(i) for i in c))))
        self.assertEqual(C.mean(), statistics.mean(c))
        self.assertEqual(C.median(), statistics.median(c))

        self.assertEqual(
            list(C.cummin()), [min(c[:i]) for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C.cummax()), [max(c[:i]) for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C.cumsum()), [sum(c[:i]) for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C.cumprod()),
            [functools.reduce(operator.mul, c[:i], 1) for i in range(1, len(c) + 1)]
            + [None],
        )
        self.assertEqual((C % 2 == 0)[:-1].all(), all(i % 2 == 0 for i in c))
        self.assertEqual((C % 2 == 0)[:-1].any(), any(i % 2 == 0 for i in c))

    def base_test_in_nunique(self):
        c = [1, 4, 2, 7]
        C = self.ts.Column(c + [None])
        self.assertEqual(list(C.isin([1, 2, 3])), [True, False, True, False, False])
        C = C.append(c)
        d = set(c)
        d.add(None)
        self.assertEqual(C.nunique(), len(set(C) - {None}))
        self.assertEqual(C.nunique(dropna=False), len(set(C)))

        self.assertEqual(C.is_unique(), False)
        self.assertEqual(self.ts.Column([1, 2, 3]).is_unique(), True)

        self.assertEqual(self.ts.Column([1, 2, 3]).is_monotonic_increasing(), True)
        self.assertEqual(self.ts.Column(dtype=dt.int64).is_monotonic_decreasing(), True)

    def base_test_math_ops(self):
        c = [1.0, 4.2, 2, 7, -9, -2.5]
        C = self.ts.Column(c + [None])

        self.assertEqual(list(C.abs()), [abs(i) for i in c] + [None])
        self.assertEqual(list(C.ceil()), [ceil(i) for i in c] + [None])
        self.assertEqual(list(C.floor()), [floor(i) for i in c] + [None])

        self.assertEqual(list(C.round()), [round(i) for i in c] + [None])
        self.assertEqual(list(C.round(2)), [round(i, 2) for i in c] + [None])
        # self.assertEqual(list(C.hash_values()), [hash(i) for i in c] + [None])

    def base_test_describe(self):
        # requires 'implicitly' torcharrow.dataframe import DataFrame
        c = self.ts.Column([1, 2, 3])
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

    def base_test_batch_collate(self):
        c = self.ts.Column([1, 2, 3, 4, 5, 6, 7])
        # test iter
        it = c.batch(2)
        res = []
        for i in it:
            res.append(list(i))
        self.assertEqual(res, [[1, 2], [3, 4], [5, 6], [7]])
        # test collate
        it = c.batch(2)
        self.assertEqual(list(IColumn.collate(it)), [1, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":

    unittest.main()
