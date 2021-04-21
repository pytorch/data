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
import torcharrow.pytorch as tap


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

    @unittest.skipUnless(tap.available, "Requires PyTorch")
    def test_to_pytorch(self):
        import torch

        df = DataFrame(
            {
                "A": ["a", "b", "c", "d", "e"],
                "B": [[1, 2], [3, None], [4, 5], [6], [7]],
                "N_B": [[1, 2], [3, 4], None, [6], [7]],
                "C": [{1: 11}, {2: 22, 3: 33}, None, {5: 55}, {6: 66}],
                "I": [1, 2, 3, 4, 5],
                "N_I": [1, 2, 3, None, 5],
                "SS": [["a"], ["b", "bb"], ["c"], ["d", None], ["e"]],
                "DSI": [{"a": 1}, {"b": 2, "bb": 22}, {}, {"d": 4}, {}],
                "N_DII": [{1: 11}, {2: 22, 3: 33}, None, {4: 44}, {}],
                "N_ROW": Column(
                    [
                        [(1, 1.1)],
                        [(2, 2.2), (3, 3.3)],
                        [],
                        [(4, 4.4), (5, None)],
                        [(6, 6.6)],
                    ],
                    dtype=List_(
                        Struct(
                            [Field("i", Int64()), Field("f", Float32(nullable=True))]
                        )
                    ),
                ),
            }
        )
        p = df["I"][1:4].to_torch()
        self.assertEqual(p.dtype, torch.int64)
        self.assertEqual(p.tolist(), [2, 3, 4])

        p = df["N_I"][1:4].to_torch()
        self.assertEqual(p.values.dtype, torch.int64)
        # last value can be anything
        self.assertEqual(p.values.tolist()[:-1], [2, 3])
        self.assertEqual(p.presence.dtype, torch.bool)
        self.assertEqual(p.presence.tolist(), [True, True, False])

        # non nullable list with nullable elements
        p = df["B"][1:4].to_torch()
        self.assertEqual(p.values.values.dtype, torch.int64)
        self.assertEqual(p.values.presence.dtype, torch.bool)
        self.assertEqual(p.offsets.dtype, torch.int32)
        self.assertEqual(p.values.values.tolist(), [3, 0, 4, 5, 6])
        self.assertEqual(p.values.presence.tolist(), [True, False, True, True, True])
        self.assertEqual(p.offsets.tolist(), [0, 2, 4, 5])

        # nullable list with non nullable elements
        p = df["N_B"][1:4].to_torch()
        self.assertEqual(p.values.values.dtype, torch.int64)
        self.assertEqual(p.presence.dtype, torch.bool)
        self.assertEqual(p.values.offsets.dtype, torch.int32)
        self.assertEqual(p.values.values.tolist(), [3, 4, 6])
        self.assertEqual(p.presence.tolist(), [True, False, True])
        self.assertEqual(p.values.offsets.tolist(), [0, 2, 2, 3])

        # list of strings -> we skip PackedList all together
        p = df["SS"][1:4].to_torch()
        self.assertEqual(p, [["b", "bb"], ["c"], ["d", None]])

        # map of strings - the keys turns into regular list
        p = df["DSI"][1:4].to_torch()
        self.assertEqual(p.keys, ["b", "bb", "d"])
        self.assertEqual(p.values.dtype, torch.int64)
        self.assertEqual(p.offsets.dtype, torch.int32)
        self.assertEqual(p.values.tolist(), [2, 22, 4])
        self.assertEqual(p.offsets.tolist(), [0, 2, 2, 3])

        # list of tuples
        p = df["N_ROW"][1:4].to_torch()
        self.assertEqual(p.offsets.dtype, torch.int32)
        self.assertEqual(p.offsets.tolist(), [0, 2, 2, 4])
        self.assertEqual(p.values.i.dtype, torch.int64)
        self.assertEqual(p.values.i.tolist(), [2, 3, 4, 5])
        self.assertEqual(p.values.f.presence.dtype, torch.bool)
        self.assertEqual(p.values.f.presence.tolist(), [True, True, True, False])
        self.assertEqual(p.values.f.values.dtype, torch.float32)
        np.testing.assert_almost_equal(p.values.f.values.numpy(), [2.2, 3.3, 4.4, 0.0])

        # Reverse conversion
        p = df.to_torch()
        df2 = tap.from_torch(p, dtype=df.dtype)
        self.assertEqual(df.dtype, df2.dtype)
        self.assertEqual(list(df), list(df2))


if __name__ == "__main__":
    unittest.main()
