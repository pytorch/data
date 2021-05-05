import operator
import unittest
from typing import List, Optional

# Skipping analyzing 'numpy': found module but no type hints or library stubs
import numpy as np  # type: ignore

# Skipping analyzing 'pandas': found module but no type hints or library stubs
import pandas as pd  # type: ignore
import pyarrow as pa  # type: ignore
from torcharrow import (
    Boolean,
    Column,
    DataFrame,
    DType,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List_,
    ListColumn,
    Struct,
    Field,
    Void,
    NumericalColumn,
    String,
    int64,
    string,
    is_floating,
    is_struct,
    is_list,
    is_map,
    is_string,
    from_arrow_table,
    from_arrow_array,
    from_pandas_dataframe,
    from_pandas_series,
)


# replicated here sinve we don't expose it from interop.py
def _column_without_nan(series, dtype):
    if dtype is None or is_floating(dtype):
        for i in series:
            if isinstance(i, float) and np.isnan(i):
                yield None
            else:
                yield i
    else:
        for i in series:
            yield i


class TestInterop(unittest.TestCase):
    def test_panda_series(self):

        s = pd.Series([1, 2, 3])
        self.assertEqual(list(s), list(from_pandas_series(s)))

        s = pd.Series([1.0, np.nan, 3])
        self.assertEqual(
            list(_column_without_nan(s, Float64(True))), list(from_pandas_series(s))
        )

        s = pd.Series([1, 2, 3])
        self.assertEqual(list(s), list(from_pandas_series(s, Int16(False))))

        s = pd.Series([1, 2, 3])
        t = from_pandas_series(s)
        self.assertEqual(t.dtype, Int64(True))
        self.assertEqual(list(s), list(from_pandas_series(s)))

        s = pd.Series([True, False, True])
        t = from_pandas_series(s)
        self.assertEqual(t.dtype, Boolean(True))
        self.assertEqual(list(s), list(from_pandas_series(s)))

        s = pd.Series(["a", "b", "c", "d", "e", "f", "g"])
        t = from_pandas_series(s)
        self.assertEqual(t.dtype, String(False))
        self.assertEqual(list(s), list(t))

    def test_panda_dataframes(self):

        s = pd.DataFrame({"a": [1, 2, 3]})
        self.assertEqual([(i,) for i in s["a"]], list(from_pandas_dataframe(s)))

        s = pd.DataFrame({"a": [1.0, np.nan, 3]})
        t = from_pandas_dataframe(s)
        self.assertEqual(
            [(i,) for i in list(_column_without_nan(s["a"], Float64(True)))], list(t)
        )

        s = pd.DataFrame({"a": [1, 2, 3]})
        t = from_pandas_dataframe(s, Struct([Field("a", Int16(False))]))
        self.assertEqual([(i,) for i in s["a"]], list(t))

        s = pd.DataFrame({"a": [1, 2, 3]})
        t = from_pandas_dataframe(s)
        self.assertEqual(t.dtype, Struct([Field("a", Int64(False))]))
        self.assertEqual([(i,) for i in s["a"]], list(t))

        s = pd.DataFrame({"a": [True, False, True]})
        t = from_pandas_dataframe(s)
        self.assertEqual(t.dtype, Struct([Field("a", Boolean(False))]))
        self.assertEqual([(i,) for i in s["a"]], list(t))

        s = pd.DataFrame({"a": ["a", "b", "c", "d", "e", "f", "g"]})
        t = from_pandas_dataframe(s)
        self.assertEqual(t.dtype, Struct([Field("a", String(False))]))
        self.assertEqual([(i,) for i in s["a"]], list(t))

        with self.assertRaises(KeyError):
            # KeyError: 'no matching test found for Void(nullable=True)', i.e.
            #    NULL Columns are not supported
            s = pd.DataFrame({"a": ["a"], "b": [1], "c": [None], "d": [1.0]})
            t = from_pandas_dataframe(s)

        s = pd.DataFrame({"a": ["a"], "b": [1], "c": [True], "d": [1.0]})
        t = from_pandas_dataframe(s)
        self.assertEqual(
            t,
            DataFrame(
                {
                    "a": Column(["a"]),
                    "b": Column([1]),
                    "c": Column([True]),
                    "d": Column([1.0]),
                }
            ),
        )

    def test_arrow_array(self):

        s = pa.array([1, 2, 3])
        t = from_arrow_array(s)
        self.assertEqual([i.as_py() for i in s], list(t))

        s = pa.array([1.0, np.nan, 3])
        t = from_arrow_array(s)
        self.assertEqual(t.dtype, Float64(False))
        # can't compare nan, so

        for i, j in zip([i.as_py() for i in s], list(t)):
            if np.isnan(i) and np.isnan(j):
                pass
            else:
                self.assertEqual(i, j)

        s = pa.array([1.0, None, 3])
        t = from_arrow_array(s)
        self.assertEqual(t.dtype, Float64(True))
        self.assertEqual([i.as_py() for i in s], list(t))

        s = pa.array([1, 2, 3], type=pa.uint32())
        self.assertEqual(
            [i.as_py() for i in s], list(from_arrow_array(s, Int16(False)))
        )

        s = pa.array([1, 2, 3])
        t = from_arrow_array(s)
        self.assertEqual(t.dtype, Int64(False))
        self.assertEqual([i.as_py() for i in s], list(from_arrow_array(s)))

        s = pa.array([True, False, True])
        t = from_arrow_array(s)
        self.assertEqual(t.dtype, Boolean(False))
        self.assertEqual([i.as_py() for i in s], list(from_arrow_array(s)))

        s = pa.array(["a", "b", "c", "d", "e", "f", "g"])
        t = from_arrow_array(s)
        self.assertEqual(t.dtype, String(False))
        self.assertEqual([i.as_py() for i in s], list(t))

    def test_arrow_table(self):

        s = pa.table({"a": [1, 2, 3]})
        t = from_arrow_table(s)
        self.assertEqual([(i.as_py(),) for i in s["a"]], list(t))

        s = pa.table({"a": [1.0, np.nan, 3]})
        t = from_arrow_table(s)

        self.assertEqual(t.dtype, Struct([Field("a", Float64(False))]))
        # can't compare nan, so

        for i, j in zip([i.as_py() for i in s["a"]], list(t["a"])):
            if np.isnan(i) and np.isnan(j):
                pass
            else:
                self.assertEqual(i, j)

        s = pa.table({"a": [1, 2, 3]})
        t = from_arrow_table(s, Struct([Field("a", Int16(False))]))
        self.assertEqual([(i.as_py(),) for i in s["a"]], list(t))

        s = pa.table({"a": [1, 2, 3]})
        t = from_arrow_table(s)
        self.assertEqual(t.dtype, Struct([Field("a", Int64(False))]))
        self.assertEqual([(i.as_py(),) for i in s["a"]], list(t))

        s = pa.table({"a": [True, False, True]})
        t = from_arrow_table(s)
        self.assertEqual(t.dtype, Struct([Field("a", Boolean(False))]))
        self.assertEqual([(i.as_py(),) for i in s["a"]], list(t))

        s = pa.table({"a": ["a", "b", "c", "d", "e", "f", "g"]})
        t = from_arrow_table(s)
        self.assertEqual(t.dtype, Struct([Field("a", String(False))]))
        self.assertEqual([(i.as_py(),) for i in s["a"]], list(t))

        with self.assertRaises(KeyError):
            # KeyError: 'no matching test found for Void(nullable=True)', i.e.
            #    NULL Columns are not supported
            s = pa.table({"a": ["a"], "b": [1], "c": [None], "d": [1.0]})
            t = from_arrow_table(s)

        s = pa.table({"a": ["a"], "b": [1], "c": [True], "d": [1.0]})
        t = from_arrow_table(s)
        self.assertEqual(
            t,
            DataFrame(
                {
                    "a": Column(["a"]),
                    "b": Column([1]),
                    "c": Column([True]),
                    "d": Column([1.0]),
                }
            ),
        )

    def test_to_python(self):
        df = DataFrame(
            {
                "A": ["a", "b", "c", "d"],
                "B": [[1, 2], [3, None], [4, 5], [6]],
                "C": [{1: 11}, {2: 22, 3: 33}, None, {5: 55}],
            }
        )
        p = df[1:3].to_python()
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0].A, "b")
        self.assertEqual(p[1].A, "c")
        self.assertEqual(p[0].B, [3, None])
        self.assertEqual(p[1].B, [4, 5])
        self.assertEqual(p[0].C, {2: 22, 3: 33})
        self.assertEqual(p[1].C, None)


if __name__ == "__main__":
    unittest.main()
