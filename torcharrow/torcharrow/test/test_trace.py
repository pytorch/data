import operator
import unittest

import torcharrow.dtypes as dt
from torcharrow import IColumn, Scope, me, trace, GroupedDataFrame

# -----------------------------------------------------------------------------
# testdata


class DF:
    @trace
    def __init__(self, session):
        self._scope = session
        self.value = session.ct.next()
        self.id = f"c{self.value}"

    def __str__(self):
        return f"DF({self._scope.ct.value})"

    @staticmethod
    @trace
    def make(session):
        return DF(session)

    @trace
    def f(self, other):
        assert not isinstance(other, DF)
        res = DF(self._scope)
        res.value = self.value * 10
        return res

    @trace
    def g(self, other):
        assert isinstance(other, DF)
        res = DF(self._scope)
        res.value = self.value * 100 + other.value * 1000
        return res


# -----------------------------------------------------------------------------
# tests


class TestDFNoTrace(unittest.TestCase):
    def setUp(self):
        self.ts = Scope()

    def test_tracing_off(self):
        trace = self.ts.trace

        self.assertTrue(not trace.is_on())
        self.assertEqual(len(trace._trace), 0)
        df1 = DF.make(self.ts)
        self.assertTrue(len(trace._trace) == 0)


class TestDFWithTrace(unittest.TestCase):
    def setUp(self, tracing=True):
        self.ts = Scope({"tracing": tracing, "types_to_trace": [Scope, DF]})

    def test_tracing_on(self):
        trace = self.ts.trace

        self.assertEqual(len(trace._trace), 0)

        df1 = DF.make(self.ts)
        df2 = DF(self.ts)
        df3 = df1.f(13)
        df4 = df3.g(df2)

        self.assertTrue(len(trace._trace) > 0)

        verdict = [
            "c0 = torcharrow.test.test_trace.DF.make(s0)",
            "c1 = DF(s0)",
            "c2 = torcharrow.test.test_trace.DF.f(c0, 13)",
            "c3 = torcharrow.test.test_trace.DF.g(c2, c1)",
        ]
        global running_locally
        if running_locally:
            for i in range(len(verdict)):
                verdict[i] = verdict[i].replace("torcharrow.test.test_trace.", "")

        self.assertEqual(trace.statements(), verdict)

    def test_trace_equivalence(self):
        trace = self.ts.trace

        df1 = DF.make(self.ts)
        df2 = DF(self.ts)
        df3 = df1.f(13)
        df4 = df3.g(df2)

        stms = trace.statements()
        result = trace.result()

        self.setUp(tracing=False)
        # s0 must bind session object, can be the same...
        import torcharrow

        s0 = self.ts
        exec(";".join(stms))
        self.assertEqual(df4.value, eval(result).value)

    def test_trace_stable(self):
        trace = self.ts.trace

        df1 = DF.make(self.ts)
        df2 = DF(self.ts)
        df3 = df1.f(13)
        df4 = df3.g(df2)

        original_stms = trace.statements()

        # redo trace; have to bind a session object under the name s0.
        import torcharrow

        s0 = Scope({"tracing": True, "types_to_trace": [Scope, DF]})
        exec(";".join(original_stms))

        traced_stms = s0.trace.statements()

        self.assertEqual(original_stms, traced_stms)


# aux: global function --------------------------------------------------------


def h(x):
    return 133 if x == 13 else x


def cmds(stms):

    quote = lambda x: f'"{x}"' if "'" in x else f"'{x}'"
    res = []
    for stm in stms:
        res.append(quote(stm))
    return "[\n\t" + "\n\t,".join(res) + "\n\t]"


# tests continued --------------------------------------------------------------


class TestColumnTrace(unittest.TestCase):
    def setUp(self):
        self.ts = Scope(
            {
                "device": "std",
                "tracing": True,
                "types_to_trace": [Scope, IColumn],
            }
        )

    def test_columns(self):

        c0 = self.ts.Column(dt.int64)
        c0 = c0.append([13])
        t = c0.dtype
        c0 = c0.append([14])

        # simply list all operations and see what happens...

        c0 = c0.append([16, 19])
        _ = c0.count()
        _ = len(c0)
        # _ = c0.ndim
        # _ = c0.size
        # TODO: do we need copy...
        c1 = c0  # .copy()
        _ = c1.get(0, None)
        s = 0
        for i in c1:
            s += i
        _ = c1[0]
        c2 = c1[:1]
        c3 = c1[[0, 1]]

        # NOTE can't be traced...
        b = self.ts.Column([True] * len(c3))
        # ... rewrite to
        b = c1 != c1

        c4 = c1[b]
        c5 = c1.head(17)
        c6 = c3.tail(-12)
        c7 = c1.map({13: 133}, **{"dtype": dt.Int64(True)})

        # # NOTE can't be traced...
        # c8 = c1.map(lambda x: 133 if x == 13 else x)

        # # ...rewrite to (must be global function, no captured vars);
        # #def h(x): return 133 if x == 13 else x

        c9 = c1.map(h)
        _ = c1.reduce(operator.add, 0)
        c10 = c0.sort(ascending=False)
        c11 = c10.nlargest()

        # print("TRACE", cmds(self.ts.trace.statements()))
        verdict = [
            "c0 = torcharrow.scope.Scope.Column(s0, int64)",
            "c2 = torcharrow.numpy_rt.numerical_column_std.NumericalColumnStd.append(c0, [13])",
            "c4 = torcharrow.numpy_rt.numerical_column_std.NumericalColumnStd.append(c2, [14])",
            "c6 = torcharrow.numpy_rt.numerical_column_std.NumericalColumnStd.append(c4, [16, 19])",
            "_ = torcharrow.icolumn.IColumn.count(c6)",
            "_ = torcharrow.icolumn.IColumn.__getitem__(c6, 0)",
            "c7 = torcharrow.icolumn.IColumn.__getitem__(c6, slice(None, 1, None))",
            "c8 = torcharrow.icolumn.IColumn.__getitem__(c6, [0, 1])",
            "c9 = torcharrow.scope.Scope.Column(s0, [True, True])",
            "c10 = torcharrow.numpy_rt.numerical_column_std.NumericalColumnStd.__ne__(c6, c6)",
            "c11 = torcharrow.icolumn.IColumn.__getitem__(c6, c10)",
            "c12 = torcharrow.icolumn.IColumn.head(c6, 17)",
            "c13 = torcharrow.icolumn.IColumn.tail(c8, -12)",
            "c14 = torcharrow.icolumn.IColumn.map(c6, {13: 133}, dtype=Int64(nullable=True))",
            "c15 = torcharrow.icolumn.IColumn.map(c6, h)",
            "_ = torcharrow.icolumn.IColumn.reduce(c6, operator.add, 0)",
            "c16 = torcharrow.numpy_rt.numerical_column_std.NumericalColumnStd.sort(c6, ascending=False)",
            "c18 = torcharrow.numpy_rt.numerical_column_std.NumericalColumnStd.nlargest(c16)",
        ]

        self.assertEqual(self.ts.trace.statements(), verdict)


# # aux: global function --------------------------------------------------------


def add(tup):
    a, b, c = tup
    return a + b


# # tests continued --------------------------------------------------------------


class TestDataframeTrace(unittest.TestCase):
    def setUp(self):
        types = [Scope, IColumn, GroupedDataFrame]
        self.ts = Scope({"device": "std", "tracing": True, "types_to_trace": types})

    def test_simple_df_ops_fail(self):

        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]

        c1 = df["a"]
        df["d"] = c1

        d1 = df.drop(["a"])
        d2 = df.keep(["a", "c"])
        # TODO Clarify: Why did we have in 0.2 this as an  self.assertRaises(AttributeError):
        # AttributeError: cannot override existing column d
        # simply overrides the column name, but that's ok...
        d3 = df.rename({"c": "d"})
        d3 = df.rename({"c": "e"})
        self.assertTrue(True)

    def test_simple_df_ops_succeed(self):

        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]

        c1 = df["a"]
        df["d"] = c1

        d1 = df.drop(["a"])
        d2 = df.keep(["a", "c"])
        d3 = d2.rename({"c": "e"})

        d4 = d3.min()

        # print("TRACE", cmds(self.ts.trace.statements()))
        verdict = [
            "c0 = torcharrow.scope.Scope.DataFrame(s0)",
            "_ = torcharrow.numpy_rt.dataframe_std.DataFrameStd.__setitem__(c0, 'a', [1, 2, 3])",
            "_ = torcharrow.numpy_rt.dataframe_std.DataFrameStd.__setitem__(c0, 'b', [11, 22, 33])",
            "_ = torcharrow.numpy_rt.dataframe_std.DataFrameStd.__setitem__(c0, 'c', [111, 222, 333])",
            "c1 = torcharrow.icolumn.IColumn.__getitem__(c0, 'a')",
            "_ = torcharrow.numpy_rt.dataframe_std.DataFrameStd.__setitem__(c0, 'd', c1)",
            "c4 = torcharrow.numpy_rt.dataframe_std.DataFrameStd.drop(c0, ['a'])",
            "c5 = torcharrow.numpy_rt.dataframe_std.DataFrameStd.keep(c0, ['a', 'c'])",
            "c6 = torcharrow.numpy_rt.dataframe_std.DataFrameStd.rename(c5, {'c': 'e'})",
            "c9 = torcharrow.numpy_rt.dataframe_std.DataFrameStd.min(c6)",
        ]

        self.assertEqual(self.ts.trace.statements(), verdict)

    def test_df_trace_equivalence(self):
        df = self.ts.DataFrame()
        self.assertEqual(
            self.ts.trace.statements(), ["c0 = torcharrow.scope.Scope.DataFrame(s0)"]
        )

        df["a"] = [1, 2, 3]
        self.assertEqual(
            self.ts.trace.statements()[-1],
            "_ = torcharrow.numpy_rt.dataframe_std.DataFrameStd.__setitem__(c0, 'a', [1, 2, 3])",
        )

        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]
        d11 = df.where((me["a"] > 1))

        # print("TRACE", cmds(self.ts.trace.statements()))
        verdict = [
            "c0 = torcharrow.scope.Scope.DataFrame(s0)",
            "_ = torcharrow.numpy_rt.dataframe_std.DataFrameStd.__setitem__(c0, 'a', [1, 2, 3])",
            "_ = torcharrow.numpy_rt.dataframe_std.DataFrameStd.__setitem__(c0, 'b', [11, 22, 33])",
            "_ = torcharrow.numpy_rt.dataframe_std.DataFrameStd.__setitem__(c0, 'c', [111, 222, 333])",
            "c8 = torcharrow.numpy_rt.dataframe_std.DataFrameStd.where(c0, torcharrow.idataframe.me.__getitem__('a').__gt__(1))",
        ]

        self.assertEqual(self.ts.trace.statements(), verdict)

        # capture trace
        stms = self.ts.trace.statements()
        result = self.ts.trace.result()

        # Pick an arbitrary Scope object; run trace
        import torcharrow

        s0 = Scope.default
        exec(";".join(stms))

        # check for equivalence
        self.assertEqual(list(d11), list(eval(result)))

    def test_df_trace_locals_and_me_equivalence(self):

        d0 = self.ts.DataFrame(
            {"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]}
        )

        d1 = d0.where((d0["a"] > 1))
        d1_result = self.ts.trace.result()

        d2 = d0.where((me["a"] > 1))
        d2_result = self.ts.trace.result()

        stms = self.ts.trace.statements()
        self.assertEqual(list(d1), list(d2))

        # restart and run trace
        import torcharrow

        s0 = Scope.default
        exec(";".join(stms))
        # check for equivalence
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))

    def test_df_trace_select_with_map(self):

        d0 = self.ts.DataFrame(
            {"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]}
        )
        d2 = d0.select(f=me.map(add, dtype=dt.int64))

        d2_result = self.ts.trace.result()
        stms = self.ts.trace.statements()

        import torcharrow
        from torcharrow.dtypes import int64

        s0 = Scope.default
        exec(";".join(stms))
        self.assertEqual(list(d2), list(eval(d2_result)))

    def test_df_trace_select_map_equivalence(self):

        d0 = self.ts.DataFrame(
            {"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]}
        )

        d1 = d0.select("*", e=me["a"] + me["b"])
        d1_result = self.ts.trace.result()

        d2 = d0.select("*", e=me.map(add, dtype=dt.int64))
        d2_result = self.ts.trace.result()

        stms = self.ts.trace.statements()
        # print("TRACE", stms)
        import torcharrow
        from torcharrow.dtypes import int64

        s0 = Scope.default
        exec(";".join(stms))
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))

    def test_df_without_input(self):
        d0 = self.ts.DataFrame(
            dtype=dt.Struct([dt.Field(i, dt.int64) for i in ["a", "b", "c"]])
        )

        d1 = d0.select("*", e=me["a"] + me["b"])
        d1_result = self.ts.trace.result()

        d2 = d0.select("*", e=me.map(add, dtype=dt.int64))
        d2_result = self.ts.trace.result()

        stms = self.ts.trace.statements()

        import torcharrow
        from torcharrow.dtypes import Struct, Field, int64

        s0 = Scope.default
        exec(";".join(stms))
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))


running_locally = False
if __name__ == "__main__":
    running_locally = True
    unittest.main()
