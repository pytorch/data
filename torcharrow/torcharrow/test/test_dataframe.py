import functools
import statistics
import unittest

import torcharrow.dtypes as dt
from torcharrow import IDataFrame, Scope, me

# run python3 -m unittest outside this directory to run all tests


class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.ts = Scope()

    def test_internals_empty(self):
        empty = self.ts.DataFrame()

        # testing internals...
        self.assertTrue(isinstance(empty, IDataFrame))

        self.assertEqual(empty.length(), 0)
        self.assertEqual(empty.null_count(), 0)
        self.assertEqual(len(empty._field_data), 0)
        self.assertEqual(len(empty._mask), 0)
        self.assertEqual(empty.columns, [])

    def test_internals_full(self):
        df = self.ts.DataFrame(dt.Struct([dt.Field("a", dt.int64)]))
        for i in range(4):
            df = df.append([(i,)])

        for i in range(4):
            self.assertEqual(df[i], (i,))

        self.assertEqual(df.length(), 4)
        self.assertEqual(df.null_count(), 0)
        self.assertEqual(len(df._field_data), 1)
        self.assertEqual(len(df._mask), 4)
        self.assertEqual(list(df), list((i,) for i in range(4)))
        m = df[0 : len(df)]
        self.assertEqual(list(df[0 : len(df)]), list((i,) for i in range(4)))
        # TODO enforce runtime type check!
        # with self.assertRaises(TypeError):
        #     # TypeError: a tuple of type dt.Struct([dt.Field(a, dt.int64)]) is required, got None
        #     df=df.append([None])
        #     self.assertEqual(df.length(), 5)
        #     self.assertEqual(df.null_count(), 1)

    def test_internals_full_nullable(self):
        with self.assertRaises(TypeError):
            #  TypeError: nullable structs require each field (like a) to be nullable as well.
            df = self.ts.DataFrame(
                dt.Struct(
                    [dt.Field("a", dt.int64), dt.Field("b", dt.int64)], nullable=True
                )
            )
        df = self.ts.DataFrame(
            dt.Struct(
                [dt.Field("a", dt.int64.with_null()), dt.Field("b", dt.Int64(True))],
                nullable=True,
            )
        )

        for i in [0, 1, 2]:
            df = df.append([None])
            # None: since dt.Struct is nullable, we add Null to the column.
            self.assertEqual(df._field_data["a"][-1], None)
            self.assertEqual(df._field_data["b"][-1], None)
            # but all public APIs report this back as None

            self.assertEqual(df[i], None)
            self.assertEqual(df.valid(i), False)
            self.assertEqual(df.null_count(), i + 1)
        for i in [3]:
            df = df.append([(i, i * i)])
            self.assertEqual(df[i], (i, i * i))
            self.assertEqual(df.valid(i), True)

        self.assertEqual(df.length(), 4)
        self.assertEqual(df.null_count(), 3)
        self.assertEqual(len(df["a"]), 4)
        self.assertEqual(len(df["b"]), 4)
        self.assertEqual(len(df._mask), 4)

        self.assertEqual(list(df), [None, None, None, (3, 9)])

        df = df.append([(4, 4 * 4), (5, 5 * 5)])
        self.assertEqual(list(df), [None, None, None, (3, 9), (4, 16), (5, 25)])

        # len
        self.assertEqual(len(df), 6)

    def test_internals_column_indexing(self):
        df = self.ts.DataFrame()
        df["a"] = self.ts.Column([None] * 3, dtype=dt.Int64(nullable=True))
        df["b"] = self.ts.Column([1, 2, 3])
        df["c"] = self.ts.Column([1.1, 2.2, 3.3])

        # index
        self.assertEqual(list(df["a"]), [None] * 3)
        # pick & column -- note: creates a view
        self.assertEqual(df[["a", "c"]].columns, ["a", "c"])
        # pick and index
        self.assertEqual(list(df[["a", "c"]]["a"]), [None] * 3)
        self.assertEqual(list(df[["a", "c"]]["c"]), [1.1, 2.2, 3.3])

        # slice

        self.assertEqual(df[:"b"].columns, ["a"])
        self.assertEqual(df["b":].columns, ["b", "c"])
        self.assertEqual(df["a":"c"].columns, ["a", "b"])

    def test_infer(self):
        df = self.ts.DataFrame({"a": [1, 2, 3], "b": [1.0, None, 3]})
        self.assertEqual(df.columns, ["a", "b"])
        self.assertEqual(
            df.dtype,
            dt.Struct(
                [dt.Field("a", dt.int64), dt.Field("b", dt.Float64(nullable=True))]
            ),
        )

        self.assertEqual(df.dtype.get("a"), dt.int64)
        self.assertEqual(list(df), list(zip([1, 2, 3], [1.0, None, 3])))

        df = self.ts.DataFrame()
        self.assertEqual(len(df), 0)

        df["a"] = self.ts.Column([1, 2, 3], dtype=dt.int32)
        self.assertEqual(df._dtype.get("a"), dt.int32)
        self.assertEqual(len(df), 3)

        df["b"] = [1.0, None, 3]
        self.assertEqual(len(df), 3)

        df = self.ts.DataFrame([(1, 2), (2, 3), (4, 5)], columns=["a", "b"])
        self.assertEqual(list(df), [(1, 2), (2, 3), (4, 5)])

        B = dt.Struct([dt.Field("b1", dt.int64), dt.Field("b2", dt.int64)])
        A = dt.Struct([dt.Field("a", dt.int64), dt.Field("b", B)])
        df = self.ts.DataFrame([(1, (2, 22)), (2, (3, 33)), (4, (5, 55))], dtype=A)

        self.assertEqual(list(df), [(1, (2, 22)), (2, (3, 33)), (4, (5, 55))])

    @staticmethod
    def _add(a, b):
        return a + b

    def test_map_where_filter(self):
        # TODO have to decide on whether to follow Pandas, map, filter or our own.

        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = ["a", "b", "C"]
        df["d"] = [100, 200, None]

        # keep None
        self.assertEqual(
            list(df.map({100: 1000}, columns=["d"], dtype=dt.Int64(nullable=True))),
            [1000, None, None],
        )

        # maps None
        self.assertEqual(
            list(
                df.map(
                    {None: 1, 100: 1000}, columns=["d"], dtype=dt.Int64(nullable=True)
                )
            ),
            [1000, None, 1],
        )

        # maps as function
        self.assertEqual(
            list(df.map(TestDataFrame._add, columns=["a", "a"], dtype=dt.int64)),
            [2, 4, 6],
        )

        # filter
        self.assertEqual(
            list(df.filter(str.islower, columns=["c"])),
            [(1, 11, "a", 100), (2, 22, "b", 200)],
        )

    def test_sort_stuff(self):
        df = self.ts.DataFrame({"a": [1, 2, 3], "b": [1.0, None, 3]})
        self.assertEqual(
            list(df.sort(by="a", ascending=False)),
            list(zip([3, 2, 1], [3, None, 1.0])),
        )
        # Not allowing None in comparison might be too harsh...
        # TODO CLARIFY THIS
        # with self.assertRaises(TypeError):
        #     # TypeError: '<' not supported between instances of 'NoneType' and 'float'
        #     self.assertEqual(
        #         list(df.sort(by="b", ascending=False)),
        #         list(zip([3, 2, 1], [3, None, 1.0])),
        #     )

        df = self.ts.DataFrame({"a": [1, 2, 3], "b": [1.0, None, 3], "c": [4, 4, 1]})
        self.assertEqual(
            list(df.sort(by=["c", "a"], ascending=False)),
            list([(2, None, 4), (1, 1.0, 4), (3, 3.0, 1)]),
        )

        self.assertEqual(
            list(df.nlargest(n=2, columns=["c", "a"], keep="first")),
            [(2, None, 4), (1, 1.0, 4)],
        )
        self.assertEqual(
            list(df.nsmallest(n=2, columns=["c", "a"], keep="first")),
            [(3, 3.0, 1), (1, 1.0, 4)],
        )
        self.assertEqual(list(df.reverse()), [(3, 3.0, 1), (2, None, 4), (1, 1.0, 4)])

    def test_operators(self):
        # without None
        c = self.ts.DataFrame({"a": [0, 1, 3]})

        d = self.ts.DataFrame({"a": [5, 5, 6]})
        e = self.ts.DataFrame({"a": [1.0, 1, 7]})

        self.assertEqual(list(c == c), [(True,)] * 3)
        self.assertEqual(list(c == d), [(False,)] * 3)

        # NOTE: Yoo cannot compare Columns with assertEqual,
        #       since torcharrow overrode __eq__
        #       this always compare with list(), etc
        #       or write (a==b).all()

        self.assertEqual(list(c == 1), [(i,) for i in [False, True, False]])
        self.assertTrue(
            ((c == 1) == self.ts.DataFrame({"a": [False, True, False]})).all()
        )

        # <, <=, >=, >

        self.assertEqual(list(c <= 2), [(i,) for i in [True, True, False]])
        self.assertEqual(list(c < d), [(i,) for i in [True, True, True]])
        self.assertEqual(list(c >= d), [(i,) for i in [False, False, False]])
        self.assertEqual(list(c > 2), [(i,) for i in [False, False, True]])

        # +,-,*,/,//,**

        self.assertEqual(list(-c), [(i,) for i in [0, -1, -3]])
        self.assertEqual(list(+-c), [(i,) for i in [0, -1, -3]])

        self.assertEqual(list(c + 1), [(i,) for i in [1, 2, 4]])
        # self.assertEqual(list(c.add(1)), [(i,) for i in [1, 2, 4]])

        self.assertEqual(list(1 + c), [(i,) for i in [1, 2, 4]])
        # self.assertEqual(list(c.radd(1)), [(i,) for i in [1, 2, 4]])

        self.assertEqual(list(c + d), [(i,) for i in [5, 6, 9]])
        # self.assertEqual(list(c.add(d)), [(i,) for i in [5, 6, 9]])

        self.assertEqual(list(c + 1), [(i,) for i in [1, 2, 4]])
        self.assertEqual(list(1 + c), [(i,) for i in [1, 2, 4]])
        self.assertEqual(list(c + d), [(i,) for i in [5, 6, 9]])

        self.assertEqual(list(c - 1), [(i,) for i in [-1, 0, 2]])
        self.assertEqual(list(1 - c), [(i,) for i in [1, 0, -2]])
        self.assertEqual(list(d - c), [(i,) for i in [5, 4, 3]])

        self.assertEqual(list(c * 2), [(i,) for i in [0, 2, 6]])
        self.assertEqual(list(2 * c), [(i,) for i in [0, 2, 6]])
        self.assertEqual(list(c * d), [(i,) for i in [0, 5, 18]])

        self.assertEqual(list(c * 2), [(i,) for i in [0, 2, 6]])
        self.assertEqual(list(2 * c), [(i,) for i in [0, 2, 6]])
        self.assertEqual(list(c * d), [(i,) for i in [0, 5, 18]])

        self.assertEqual(list(c / 2), [(i,) for i in [0.0, 0.5, 1.5]])
        # #TODO check numpy devision issue
        # with self.assertRaises(ZeroDivisionError):
        #     self.assertEqual(list(2 / c), [(i,) for i in [0.0, 0.5, 1.5]])
        self.assertEqual(list(c / d), [(i,) for i in [0.0, 0.2, 0.5]])

        self.assertEqual(list(d // 2), [(i,) for i in [2, 2, 3]])
        self.assertEqual(list(2 // d), [(i,) for i in [0, 0, 0]])
        self.assertEqual(list(c // d), [(i,) for i in [0, 0, 0]])
        self.assertEqual(list(e // d), [(i,) for i in [0.0, 0.0, 1.0]])
        # THIS ASSERTION SHOULD NOT HAPPEN, FIX derive_dtype
        # TypeError: integer argument expected, got float
        # TODO check numpy devision issue
        # with self.assertRaises(TypeError):
        #     self.assertEqual(list(d // e), [(i,) for i in [0.0, 0.0, 1.0]])

        self.assertEqual(list(c ** 2), [(i,) for i in [0, 1, 9]])
        self.assertEqual(list(2 ** c), [(i,) for i in [1, 2, 8]])
        self.assertEqual(list(c ** d), [(i,) for i in [0, 1, 729]])

        #     # # null handling

        c = self.ts.DataFrame({"a": [0, 1, 3, None]})
        self.assertEqual(list(c + 1), [(i,) for i in [1, 2, 4, None]])

        # # TODO decideo on special handling with fill_values, maybe just drop functionality?
        # self.assertEqual(list(c.add(1, fill_value=17)), [(i,) for i in [1, 2, 4, 18]])
        # self.assertEqual(list(c.radd(1, fill_value=-1)), [(i,) for i in [1, 2, 4, 0]])
        f = self.ts.Column([None, 1, 3, None])
        # self.assertEqual(
        #     list(c.radd(f, fill_value=100)), [(i,) for i in [100, 2, 6, 200]]
        # )
        self.assertEqual(list((c + f).fillna(100)), [(i,) for i in [100, 2, 6, 100]])
        # &, |, ~
        g = self.ts.Column([True, False, True, False])
        h = self.ts.Column([False, False, True, True])
        self.assertEqual(list(g & h), [False, False, True, False])
        self.assertEqual(list(g | h), [True, False, True, True])
        self.assertEqual(list(~g), [False, True, False, True])

        # Expressions should throw if anything is wrong
        try:
            u = self.ts.Column(list(range(5)))
            v = -u
            uv = self.ts.DataFrame({"a": u, "b": v})
            uu = self.ts.DataFrame({"a": u, "b": u})
            x = uv == 1
            y = uu["a"] == uv["a"]
            z = uv == uu
            z["a"]
            (z | (x["a"]))
        except:
            self.assertTrue(False)

    def test_na_handling(self):
        c = self.ts.DataFrame({"a": [None, 2, 17.0]})

        self.assertEqual(list(c.fillna(99.0)), [(i,) for i in [99.0, 2, 17.0]])
        self.assertEqual(list(c.dropna()), [(i,) for i in [2, 17.0]])

        c = c.append([(2,)])
        self.assertEqual(list(c.drop_duplicates()), [(i,) for i in [None, 2, 17.0]])

        # duplicates with subset
        d = self.ts.DataFrame({"a": [None, 2, 17.0, 7, 2], "b": [1, 2, 17.0, 2, 1]})
        self.assertEqual(
            list(d.drop_duplicates(subset="a")),
            [(None, 1.0), (2.0, 2.0), (17.0, 17.0), (7.0, 2.0)],
        )
        self.assertEqual(
            list(d.drop_duplicates(subset="b")), [(None, 1.0), (2.0, 2.0), (17.0, 17.0)]
        )
        self.assertEqual(
            list(d.drop_duplicates(subset=["b", "a"])),
            [(None, 1.0), (2.0, 2.0), (17.0, 17.0), (7.0, 2.0), (2.0, 1.0)],
        )
        self.assertEqual(
            list(d.drop_duplicates()),
            [(None, 1.0), (2.0, 2.0), (17.0, 17.0), (7.0, 2.0), (2.0, 1.0)],
        )

    def test_agg_handling(self):
        import functools
        import operator

        c = [1, 4, 2, 7, 9, 0]
        C = self.ts.DataFrame({"a": [1, 4, 2, 7, 9, 0, None]})

        self.assertEqual(C.min()["a"], min(c))
        self.assertEqual(C.max()["a"], max(c))
        self.assertEqual(C.sum()["a"], sum(c))
        self.assertEqual(C.prod()["a"], functools.reduce(operator.mul, c, 1))
        # TODO check for mode in numpy
        # self.assertEqual(C.mode()["a"], statistics.mode(c))
        self.assertEqual(C.std()["a"], statistics.stdev(c))
        self.assertEqual(C.mean()["a"], statistics.mean(c))
        self.assertEqual(C.median()["a"], statistics.median(c))

        self.assertEqual(
            list(C.cummin()),
            [(i,) for i in [min(c[:i]) for i in range(1, len(c) + 1)] + [None]],
        )
        self.assertEqual(
            list(C.cummax()),
            [(i,) for i in [max(c[:i]) for i in range(1, len(c) + 1)] + [None]],
        )
        self.assertEqual(
            list(C.cumsum()),
            [(i,) for i in [sum(c[:i]) for i in range(1, len(c) + 1)] + [None]],
        )
        self.assertEqual(
            list(C.cumprod()),
            [
                (i,)
                for i in [
                    functools.reduce(operator.mul, c[:i], 1)
                    for i in range(1, len(c) + 1)
                ]
                + [None]
            ],
        )
        self.assertEqual((C % 2 == 0)[:-1].all(), all(i % 2 == 0 for i in c))
        self.assertEqual((C % 2 == 0)[:-1].any(), any(i % 2 == 0 for i in c))

    def test_isin(self):
        c = [1, 4, 2, 7]
        C = self.ts.DataFrame({"a": c + [None]})
        self.assertEqual(
            list(C.isin([1, 2, 3])), [(i,) for i in [True, False, True, False, False]]
        )

    def test_isin2(self):
        df = self.ts.DataFrame({"A": [1, 2, 3], "B": [1, 1, 1]})
        self.assertEqual(list(df.nunique()), [("A", 3), ("B", 1)])

    def test_describe_dataframe(self):
        # TODO introduces cyclic dependency between Column and Dataframe, need diff design...
        c = self.ts.DataFrame({"a": self.ts.Column([1, 2, 3])})
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

    def test_drop_keep_rename_reorder_pipe(self):
        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]
        self.assertEqual(list(df.drop([])), [(1, 11, 111), (2, 22, 222), (3, 33, 333)])
        self.assertEqual(list(df.drop(["c", "a"])), [(11,), (22,), (33,)])

        self.assertEqual(list(df.keep([])), [])
        self.assertEqual(list(df.keep(["c", "a"])), [(1, 111), (2, 222), (3, 333)])

        self.assertEqual(
            list(df.rename({"a": "c", "c": "a"})),
            [(1, 11, 111), (2, 22, 222), (3, 33, 333)],
        )
        self.assertEqual(
            list(df.reorder(list(reversed(df.columns)))),
            [(111, 11, 1), (222, 22, 2), (333, 33, 3)],
        )

        def f(df):
            return df

        self.assertEqual(list(df), list(df.pipe(f)))

        def g(df, num):
            return df + num

        self.assertEqual(list(df + 13), list(df.pipe(g, 13)))

    def test_me_on_str(self):
        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = ["a", "b", "C"]

        self.assertEqual(
            list(df.where(me["c"].str.capitalize() == me["c"])), [(3, 33, "C")]
        )

    def test_locals_and_me_equivalence(self):
        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]

        self.assertEqual(
            list(df.where((me["a"] > 1) & (me["b"] == 33))),
            list(df[(df["a"] > 1) & (df["b"] == 33)]),
        )

        self.assertEqual(list(df.select("*")), list(df))

        self.assertEqual(list(df.select("a")), list(df.keep(["a"])))
        self.assertEqual(list(df.select("*", "-a")), list(df.drop(["a"])))

        gf = self.ts.DataFrame({"a": df["a"], "b": df["b"], "c": df["a"] + df["b"]})
        self.assertEqual(list(df.select("*", d=me["a"] + me["b"])), list(gf))

    def test_groupby_size_pipe(self):
        df = self.ts.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [2, 2, 1]})
        self.assertEqual(list(df.groupby("a").size), [(1, 2), (2, 1)])

        df = self.ts.DataFrame({"A": ["a", "b", "a", "b"], "B": [1, 2, 3, 4]})

        # TODO have to add type inference here
        # self.assertEqual(list(df.groupby('A').pipe({'B': lambda x: x.max() - x.min()})),
        #                  [('a',  2), ('b', 2)])

        # self.assertEqual(list(df.groupby('A').select(B=me['B'].max() - me['B'].min())),
        #                  [('a',  2), ('b', 2)])

    def test_groupby_agg(self):
        df = self.ts.DataFrame({"A": ["a", "b", "a", "b"], "B": [1, 2, 3, 4]})

        self.assertEqual(list(df.groupby("A").agg("sum")), [("a", 4), ("b", 6)])

        df = self.ts.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [2, 2, 1]})

        self.assertEqual(list(df.groupby("a").agg("sum")), [(1, 3, 4), (2, 3, 1)])

        self.assertEqual(
            list(df.groupby("a").agg(["sum", "min"])),
            [(1, 3, 4, 1, 2), (2, 3, 1, 3, 1)],
        )

        self.assertEqual(
            list(df.groupby("a").agg({"c": "max", "b": ["min", "mean"]})),
            [(1, 2, 1, 1.5), (2, 1, 3, 3.0)],
        )

    def test_groupby_iter_get_item_ops(self):
        df = self.ts.DataFrame({"A": ["a", "b", "a", "b"], "B": [1, 2, 3, 4]})
        for g, gf in df.groupby("A"):
            if g == ("a",):
                self.assertEqual(list(gf), [(1,), (3,)])
            elif g == ("b",):
                self.assertEqual(list(gf), [(2,), (4,)])
            else:
                self.assertTrue(False)

        self.assertEqual(list(df.groupby("A").sum()), [("a", 4), ("b", 6)])
        self.assertEqual(list(df.groupby("A")["B"].sum()), [4, 6])


if __name__ == "__main__":
    unittest.main()
