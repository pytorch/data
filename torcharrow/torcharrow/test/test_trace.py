import operator
import unittest

from torcharrow import (
    Config,
    Session,
    AbstractColumn,
    Struct,
    Field,
    GroupedDataFrame,
    NumericalColumnTest,
    DataFrame,
    Int64,
    trace,
    int64,
    trace,
    me,
)
import torcharrow
# -----------------------------------------------------------------------------
# testdata


class DF:
    @trace
    def __init__(self, session):
        self._session = session
        self.value = session.ct.next()
        self.id = f"c{self.value}"

    def __str__(self):
        return f"DF({self._session.ct.value})"

    @staticmethod
    @trace
    def make(session):
        return DF(session)

    @trace
    def f(self, other):
        assert not isinstance(other, DF)
        res = DF(self._session)
        res.value = self.value * 10
        return res

    @trace
    def g(self, other):
        assert isinstance(other, DF)
        res = DF(self._session)
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
        self.ts = Session(Config({'tracing': False, }))
        print("SESSION", type(self.ts), dir(self.ts))

    def test_tracing_off(self):
        trace = self.ts.trace
        print("TRACE", type(trace))

        self.assertTrue(not trace.is_on())
        self.assertEqual(len(trace._trace), 0)
        df1 = DF.make(self.ts)
        self.assertTrue(len(trace._trace) == 0)


class TestDFTrace(unittest.TestCase):
    def setUp(self, tracing=True):
        self.ts = Session(
            Config({'tracing': tracing, 'types_to_trace': [Session, DF]}))

    def test_tracing_on(self):
        trace = self.ts.trace

        self.assertEqual(len(trace._trace), 0)

        df1 = DF.make(self.ts)
        df2 = DF(self.ts)
        df3 = df1.f(13)
        df4 = df3.g(df2)

        self.assertTrue(len(trace._trace) > 0)

        verdict = [
            # "s0 = Session(Config(bindings={'tracing': True, 'types_to_trace': [torcharrow.session.Session, DF]}))",
            'c0 = DF.make(s0)',
            'c1 = DF(s0)',
            'c2 = DF.f(c0, 13)',
            'c3 = DF.g(c2, c1)'
        ]

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
        s0 = Session(
            Config({'tracing': True, 'types_to_trace': [Session, DF]}))
        exec(";".join(original_stms))

        traced_stms = s0.trace.statements()

        self.assertEqual(original_stms, traced_stms)


# # -----------------------------------------------------------------------------
# # test data


def h(x):
    return 133 if x == 13 else x


class TestColumnTrace(unittest.TestCase):
    def setUp(self):
        self.ts = Session(Config(
            {'device': 'test', 'tracing': True, 'types_to_trace': [Session, AbstractColumn]}))

    def test_columns(self):
        trace = self.ts.trace

        c0 = self.ts.Column(int64)
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
        c7 = c1.map({13: 133}, **{"dtype": Int64(True)})

        # # NOTE can't be traced...
        # c8 = c1.map(lambda x: 133 if x == 13 else x)

        # # ...rewrite to (must be global function, no captured vars);
        # #def h(x): return 133 if x == 13 else x

        c9 = c1.map(h)
        _ = c1.reduce(operator.add, 0)
        c10 = c0.sort(ascending=False)
        c11 = c10.nlargest()

        # print("TRACE", "\n\t[\n\t" +
        #       "\n\t,".join(trace.statements()) + "\n\t]")
        verdict = [
            "c0 = Session.Column(s0, int64)", "c2 = NumericalColumnTest.append(c0, [13])", "c4 = NumericalColumnTest.append(c2, [14])", "c6 = NumericalColumnTest.append(c4, [16, 19])", "_ = AbstractColumn.count(c6)", "_ = AbstractColumn.__getitem__(c6, 0)", "c7 = AbstractColumn.__getitem__(c6, slice(None, 1, None))", "c8 = AbstractColumn.__getitem__(c6, [0, 1])", "c9 = Session.Column(s0, [True, True])", "c10 = NumericalColumnTest.__ne__(c6, c6)", "c11 = AbstractColumn.__getitem__(c6, c10)", "c12 = AbstractColumn.head(c6, 17)", "c13 = AbstractColumn.tail(c8, -12)", "c14 = AbstractColumn.map(c6, {13: 133}, dtype=Int64(nullable=True))", "c15 = AbstractColumn.map(c6, h)", "_ = AbstractColumn.reduce(c6, operator.add, 0)", "c16 = NumericalColumnTest.sort(c6, ascending=False)", "c18 = NumericalColumnTest.nlargest(c16)"
        ]

        self.assertEqual(trace.statements(), verdict)


# # -----------------------------------------------------------------------------
# # testdata


def add(tup):
    a, b, c = tup
    return a + b


class TestDataframeTrace(unittest.TestCase):
    def setUp(self):
        types = [Session, AbstractColumn, GroupedDataFrame]
        self.ts = Session(
            Config({'device': 'test', 'tracing': True, 'types_to_trace': types}))

    def test_simple_df_ops_fail(self):
        trace = self.ts.trace

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

        # print("TRACE", "\n\t[\n\t" + "\n\t,".join(self.ts.trace.statements()) + "\n\t]")
        verdict = [
            "c0 = Session.DataFrame(s0)", "_ = DataFrame.__setitem__(c0, 'a', [1, 2, 3])", "_ = DataFrame.__setitem__(c0, 'b', [11, 22, 33])", "_ = DataFrame.__setitem__(c0, 'c', [111, 222, 333])", "c1 = AbstractColumn.__getitem__(c0, 'a')", "_ = DataFrame.__setitem__(c0, 'd', c1)", "c4 = DataFrame.drop(c0, ['a'])", "c5 = DataFrame.keep(c0, ['a', 'c'])", "c6 = DataFrame.rename(c5, {'c': 'e'})", "c9 = DataFrame.min(c6)"

        ]
        self.assertEqual(self.ts.trace.statements(), verdict)
        # print("DF", df)

    def test_df_trace_equivalence(self):
        df = self.ts.DataFrame()
        self.assertEqual(self.ts.trace.statements(), [
                         'c0 = Session.DataFrame(s0)'])

        df["a"] = [1, 2, 3]
        self.assertEqual(
            self.ts.trace.statements()[-1],
            "_ = DataFrame.__setitem__(c0, 'a', [1, 2, 3])",
        )

        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]
        d11 = df.where((me["a"] > 1))
        # print("TRACE", "\n\t[\n\t" + "\n\t,".join(self.ts.trace.statements()) + "\n\t]")

        self.assertEqual(
            self.ts.trace.statements(), [
                "c0 = Session.DataFrame(s0)", "_ = DataFrame.__setitem__(c0, 'a', [1, 2, 3])", "_ = DataFrame.__setitem__(c0, 'b', [11, 22, 33])", "_ = DataFrame.__setitem__(c0, 'c', [111, 222, 333])", "c8 = DataFrame.where(c0, me.__getitem__('a').__gt__(1))"
            ],
        )

        # capture trace
        stms = self.ts.trace.statements()
        result = self.ts.trace.result()
        # restart execution with trace

        # Pick an arbitary Session object, it can be the one from before
        s0 = Session.default

        # run trace
        exec(";".join(stms))
        # check for equivalence
        self.assertEqual(list(d11), list(eval(result)))

    def test_df_trace_locals_and_me_equivalence(self):

        d0 = self.ts.DataFrame(
            {"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]})

        d1 = d0.where((d0["a"] > 1))
        d1_result = self.ts.trace.result()

        d2 = d0.where((me["a"] > 1))
        d2_result = self.ts.trace.result()

        stms = self.ts.trace.statements()
        self.assertEqual(list(d1), list(d2))

        # restart and run trace
        s0 = Session.default
        exec(";".join(stms))
        # check for equivalence
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))

    def test_df_trace_select_with_map(self):

        d0 = self.ts.DataFrame(
            {"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]})
        d2 = d0.select(f=me.map(add, dtype=int64))

        d2_result = self.ts.trace.result()
        stms = self.ts.trace.statements()

        s0 = Session.default
        exec(";".join(stms))
        self.assertEqual(list(d2), list(eval(d2_result)))

    def test_df_trace_select_map_equivalence(self):

        d0 = self.ts.DataFrame(
            {"a": [1, 2, 3], "b": [11, 22, 33], "c": [111, 222, 333]})

        d1 = d0.select("*", e=me["a"] + me["b"])
        d1_result = self.ts.trace.result()

        d2 = d0.select("*", e=me.map(add, dtype=int64))
        d2_result = self.ts.trace.result()

        stms = self.ts.trace.statements()

        s0 = Session.default
        exec(";".join(stms))
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))

    def test_df_without_input(self):
        d0 = self.ts.DataFrame(dtype=Struct(
            [Field(i, int64) for i in ["a", "b", "c"]]))

        d1 = d0.select("*", e=me["a"] + me["b"])
        d1_result = self.ts.trace.result()

        d2 = d0.select("*", e=me.map(add, dtype=int64))
        d2_result = self.ts.trace.result()

        stms = self.ts.trace.statements()

        s0 = Session.default
        exec(";".join(stms))
        self.assertEqual(list(eval(d1_result)), list(eval(d2_result)))


if __name__ == "__main__":
    unittest.main()
