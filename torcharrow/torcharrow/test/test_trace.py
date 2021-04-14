import operator
import unittest

from torcharrow import (
    AbstractColumn,
    Column,
    DataFrame,
    Field,
    GroupedDataFrame,
    Int64,
    Struct,
    Trace,
    int64,
    me,
    trace,
)

# -----------------------------------------------------------------------------
# testdata


class DF:
    ct = 1

    @staticmethod
    def reset():
        DF.ct = 1

    def __str__(self):
        return f"DF({self.id})"

    @staticmethod
    @trace
    def make():
        return DF()

    @trace
    def __init__(self):
        self.id = f"c{DF.ct}"
        self.value = DF.ct
        DF.ct += 1

    @trace
    def f(self, other):
        assert not isinstance(other, DF)
        res = DF()
        res.value = self.value * 10
        return res

    @trace
    def g(self, other):
        assert isinstance(other, DF)
        res = DF()
        res.value = self.value * 100 + other.value * 1000
        return res


# -----------------------------------------------------------------------------
# tests
def statements(t):
    lines = t.split("\n")
    slines = map(str.strip, lines)
    flines = filter(lambda x: x is not None and len(x) > 0, slines)
    return list(flines)


class TestDFTrace(unittest.TestCase):
    def setUp(self):
        DF.reset()
        Trace.reset()

    def test_tracing_off(self):
        self.assertTrue(not Trace.is_on())
        self.assertEqual(len(Trace._trace), 0)
        df1 = DF.make()
        self.assertTrue(len(Trace._trace) == 0)

    def test_tracing_on(self):
        self.assertTrue(not Trace.is_on())
        Trace.turn(on=True, types=(DF,))
        self.assertEqual(len(Trace._trace), 0)

        df1 = DF.make()
        df2 = DF()
        df3 = df1.f(13)
        df4 = df3.g(df2)

        self.assertTrue(len(Trace._trace) > 0)

        verdict = [
            "c1 = DF.make()",
            "c2 = DF()",
            "c3 = DF.f(c1, 13)",
            "c4 = DF.g(c3, c2)",
        ]

        self.assertEqual(Trace.statements(), verdict)

    def test_trace_equivalence(self):
        Trace.turn(on=True, types=(DF,))

        df1 = DF.make()
        df2 = DF()
        df3 = df1.f(13)
        df4 = df3.g(df2)

        stms = Trace.statements()
        result = Trace.result()

        self.setUp()

        Trace.turn(on=False)
        exec(";".join(stms))
        self.assertEqual(df4.value, eval(result).value)

    def test_trace_stable(self):
        Trace.turn(on=True, types=(DF,))

        df1 = DF.make()
        df2 = DF()
        df3 = df1.f(13)
        df4 = df3.g(df2)

        original_stms = Trace.statements()

        # redo trace
        self.setUp()
        Trace.turn(on=True, types=(DF,))

        exec(";".join(original_stms))

        traced_stms = Trace.statements()
        self.assertEqual(original_stms, traced_stms)


# -----------------------------------------------------------------------------
# test data


def h(x):
    return 133 if x == 13 else x


class TestColumnTrace(unittest.TestCase):
    def setUp(self):
        AbstractColumn.reset()
        Trace.reset()

    def test_columns(self):
        Trace.turn(on=True, types=(AbstractColumn,))

        c0 = Column(int64)
        c0.append(13)
        t = c0.dtype
        c0.append(14)

        # simply list all operations and see what happens...

        c0.extend([16, 19])
        _ = c0.count()
        _ = len(c0)
        _ = c0.ndim
        _ = c0.size
        c1 = c0.copy()
        _ = c1.get(0, None)
        s = 0
        for i in c1:
            s += i
        _ = c1[0]
        c2 = c1[:1]
        c3 = c1[[0, 1]]

        # NOTE can't be traced...
        b = Column([True] * len(c3))
        # ... rewrite to
        b = c1 != c1

        c4 = c1[b]
        c5 = c1.head(17)
        c6 = c3.tail(-12)
        c7 = c1.map({13: 133}, **{"dtype": Int64(True)})

        # # NOTE can't be traced...
        # c8 = c1.map(lambda x: 133 if x == 13 else x)

        # # ...rewrite to (must be global function, no captured vars);
        # #def h(x): return 133 if x == 13 else x

        c9 = c1.map(h)
        _ = c1.reduce(operator.add, 0)
        c10 = c0.sort_values(ascending=False)
        c11 = c10.nlargest()

        verdict = """\
        c0 = Column(int64)
        _ = AbstractColumn.append(c0, 13)
        _ = c0.dtype
        _ = AbstractColumn.append(c0, 14)
        _ = AbstractColumn.count(c0)
        _ = AbstractColumn.__len__(c0)
        _ = c0.ndim
        _ = c0.size
        c1 = AbstractColumn.copy(c0)
        _ = AbstractColumn.__getitem__(c1, 0)
        c1 = AbstractColumn.__getitem__(c1, slice(None, 1, None))
        c2 = AbstractColumn.__getitem__(c1, [0, 1])
        _ = AbstractColumn.__len__(c2)
        c3 = Column([True, True])
        c4 = AbstractColumn._binary_operator(c1, 'ne', c1)
        c5 = AbstractColumn.__getitem__(c1, c4)
        c1 = AbstractColumn.head(c1, 17)
        c2 = AbstractColumn.tail(c2, -12)
        c6 = AbstractColumn.map(c1, {13: 133}, dtype=Int64(nullable=True))
        c7 = AbstractColumn.map(c1, h)
        _ = AbstractColumn.reduce(c1, operator.add, 0)
        c8 = AbstractColumn.sort_values(c0, ascending=False)
        c9 = AbstractColumn.nlargest(c8)\
        """

        self.assertEqual(Trace.statements(), statements(verdict))


# -----------------------------------------------------------------------------
# testdata


def add(tup):
    a, b, c = tup
    return a + b


class TestDataframeTrace(unittest.TestCase):
    def setUp(self):
        AbstractColumn.reset()
        Trace.reset()

    def test_simple_df_ops_fail(self):
        Trace.turn(on=True, types=(AbstractColumn, GroupedDataFrame))

        df = DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]

        c1 = df["a"]
        df["d"] = c1

        d1 = df.drop(["a"])
        d2 = df.keep(["a", "c"])
        with self.assertRaises(AttributeError):
            # AttributeError: cannot override existing column d
            d3 = df.rename({"c": "d"})
        # TODO clarify why you can't extend a trace
        d3 = df.rename({"c": "e"})
        self.assertTrue(True)

    def test_simple_df_ops_suceed(self):
        Trace.turn(on=True, types=(AbstractColumn, GroupedDataFrame))

        df = DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]

        c1 = df["a"]
        df["d"] = c1

        d1 = df.drop(["a"])
        d2 = df.keep(["a", "c"])
        d3 = d2.rename({"c": "e"})

        d4 = d3.min()

        verdict = [
            "c0 = DataFrame()",
            "_ = DataFrame.__setitem__(c0, 'a', [1, 2, 3])",
            "_ = DataFrame.__setitem__(c0, 'b', [11, 22, 33])",
            "_ = DataFrame.__setitem__(c0, 'c', [111, 222, 333])",
            "c1 = AbstractColumn.__getitem__(c0, 'a')",
            "_ = DataFrame.__setitem__(c0, 'd', c1)",
            "c4 = DataFrame.drop(c0, ['a'])",
            "c5 = DataFrame.keep(c0, ['a', 'c'])",
            "c6 = DataFrame.rename(c5, {'c': 'e'})",
            "c7 = DataFrame.min(c6)",
        ]

        self.assertEqual(Trace.statements(), verdict)

    def test_df_trace_equivalence(self):
        Trace.turn(on=True, types=(AbstractColumn, GroupedDataFrame))

        df = DataFrame()
        self.assertEqual(Trace.statements(), ["c0 = DataFrame()"])

        df["a"] = [1, 2, 3]
        self.assertEqual(
            Trace.statements(),
            ["c0 = DataFrame()", "_ = DataFrame.__setitem__(c0, 'a', [1, 2, 3])"],
        )

        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]
        d11 = df.where((me["a"] > 1))
        self.assertEqual(
            Trace.statements(),
            [
                "c0 = DataFrame()",
                "_ = DataFrame.__setitem__(c0, 'a', [1, 2, 3])",
                "_ = DataFrame.__setitem__(c0, 'b', [11, 22, 33])",
                "_ = DataFrame.__setitem__(c0, 'c', [111, 222, 333])",
                "c5 = DataFrame.where(c0, me.__getitem__('a').__gt__(1))",
            ],
        )

        # capture trace
        stms = Trace.statements()
        result = Trace.result()
        # restart execution with trace
        self.setUp()
        Trace.turn(on=False)
        # run trace
        exec(";".join(stms))
        # check for equivalence

    #     self.assertEqual(list(d11), list(eval(result)))

    def test_df_trace_locals_and_me_equivalence(self):
        Trace.turn(on=True, types=(AbstractColumn, GroupedDataFrame))
        d0 = DataFrame({"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]})

        d1 = d0.where((d0["a"] > 1))
        d1_result = Trace.result()

        d2 = d0.where((me["a"] > 1))
        d2_result = Trace.result()

        stms = Trace.statements()
        self.assertEqual(list(d1), list(d2))

        # restart
        self.setUp()
        Trace.turn(on=False)
        # run trace
        exec(";".join(stms))
        # check for equivalence
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))

    def test_df_trace_select_with_map(self):
        Trace.turn(on=True, types=(AbstractColumn, GroupedDataFrame))

        d0 = DataFrame({"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]})
        d2 = d0.select(f=me.map(add, dtype=int64))

        d2_result = Trace.result()
        stms = Trace.statements()
        self.setUp()
        Trace.turn(on=False)

        exec(";".join(stms))
        self.assertEqual(list(d2), list(eval(d2_result)))

    def test_df_trace_select_map_equivalence(self):
        Trace.turn(on=True, types=(AbstractColumn, GroupedDataFrame))
        d0 = DataFrame({"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]})

        d1 = d0.select("*", e=me["a"] + me["b"])
        d1_result = Trace.result()

        d2 = d0.select("*", e=me.map(add, dtype=int64))
        d2_result = Trace.result()

        stms = Trace.statements()

        self.setUp()
        Trace.turn(on=False)

        exec(";".join(stms))
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))

    def test_df_without_input(self):
        Trace.turn(on=True, types=(AbstractColumn, GroupedDataFrame))
        d0 = DataFrame(dtype=Struct([Field(i, int64) for i in ["a", "b", "c"]]))

        d1 = d0.select("*", e=me["a"] + me["b"])
        d1_result = Trace.result()

        d2 = d0.select("*", e=me.map(add, dtype=int64))
        d2_result = Trace.result()

        stms = Trace.statements()

        self.setUp()
        Trace.turn(on=False)

        exec(";".join(stms))
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))


if __name__ == "__main__":
    unittest.main()
