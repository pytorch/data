import operator
import unittest
from typing import List, Optional, cast

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
)


def from_arrow_table(
    table, dtype: Optional[DType] = None, columns: Optional[List[str]] = None
):
    """ "
    Convert arrow table to a torcharrow dataframe.
    """
    assert isinstance(table, pa.Table)
    if dtype is not None:
        assert is_struct(dtype)
        dtype = cast(Struct, dtype)
        res = DataFrame()
        for f in dtype.fields:
            chunked_array = table.column(f.name)
            pydata = chunked_array.to_pylist()
            res[f.name] = Column(pydata, f.dtype)
        return res
    else:
        res = DataFrame()
        table = table.select(columns) if columns is not None else table
        for n in table.column_names:
            chunked_array = table.column(n)
            pydata = chunked_array.to_pylist()
            res[n] = Column(
                pydata,
                dtype=_arrowtype_to_dtype(
                    table.schema.field(n).type, table.column(n).null_count > 0
                ),
            )
        return res


def from_pandas_dataframe(
    df, dtype: Optional[DType] = None, columns: Optional[List[str]] = None
):
    """ "
    Convert pandas dataframe to  torcharrow dataframe (drops indices).
    """
    if dtype is not None:
        assert is_struct(dtype)
        dtype = cast(Struct, dtype)
        res = DataFrame()
        for f in dtype.fields:
            # this shows that Column shoud also construct Dataframes!
            res[f.name] = _column_without_nan(df[f.name], f.dtype)
        return res
    else:
        res = DataFrame()
        for n in df.columns:
            if columns is None or n in columns:
                res[n] = _column_without_nan(
                    df[n], dtype=_pandatype_to_dtype(df[n].dtype, True)
                )

        return res


def from_arrow_array(array, dtype=None):
    """ "
    Convert arrow array to a torcharrow column.
    """
    assert isinstance(array, pa.Array)
    pydata = _arrow_scalar_to_py(array)
    if dtype is not None:
        assert not is_struct(dtype)
        return Column(pydata, dtype)
    else:
        return Column(
            pydata, dtype=_arrowtype_to_dtype(array.type, array.null_count > 0)
        )


def from_pandas_series(series, dtype=None):
    """ "
    Convert pandas series array to a torcharrow column (drops indices).
    """
    assert isinstance(series, pd.Series)
    if dtype is not None:
        assert not is_struct(dtype)
        if dtype.nullable:
            return Column(_column_without_nan(series, dtype), dtype=dtype)
        else:
            return Column(series, dtype=dtype)
    else:
        dtype = _pandatype_to_dtype(series.dtype, True)
        return Column(_column_without_nan(series, dtype), dtype=dtype)


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


def _arrow_scalar_to_py(array):
    for i in array:
        yield i.as_py()


def _pandatype_to_dtype(t, nullable):
    return _numpytype_to_dtype(t, nullable)


def _numpytype_to_dtype(t, nullable):
    if t == np.bool_:
        return Boolean(nullable)
    if t == np.int8:
        return Int8(nullable)
    if t == np.int16:
        return Int16(nullable)
    if t == np.int32:
        return Int32(nullable)
    if t == np.int64:
        return Int64(nullable)
    # if is_uint8(t): return Int8(nullable)
    # if is_uint16(t): return Int8(nullable)
    # if is_uint32(t): return Int8(nullable)
    # if is_uint64(t): return Int8(nullable)
    if t == np.float32:
        return Float32(nullable)
    if t == np.float64:
        return Float64(nullable)
    # if is_list(t):
    #     return List(t.value_type, nullable)
    if t.char == "V" and t.names is not None:
        fs = []
        for n, shape in t.fields.items():
            fs[n] = _pandatype_to_dtype(shape[0], True)
        return Struct(fs, nullable)
    # if is_null(t):
    #     return void
    if t.char == "U":  # UGLY, but...
        return String(nullable)
    # if t.char == 'O':
    #     return Map(t.item_type, t.key_type, nullable)
    if isinstance(t, object):
        return None

    raise NotImplementedError(
        f"unsupported case {t} {type(t).__name__} {nullable} {'dtype[object_]'==type(t).__name__}"
    )


def _arrowtype_to_dtype(t, nullable):
    if pa.types.is_boolean(t):
        return Boolean(nullable)
    if pa.types.is_int8(t):
        return Int8(nullable)
    if pa.types.is_int16(t):
        return Int16(nullable)
    if pa.types.is_int32(t):
        return Int32(nullable)
    if pa.types.is_int64(t):
        return Int64(nullable)
    # if is_uint8(t): return Int8(nullable)
    # if is_uint16(t): return Int8(nullable)
    # if is_uint32(t): return Int8(nullable)
    # if is_uint64(t): return Int8(nullable)
    if pa.types.is_float32(t):
        return Float32(nullable)
    if pa.types.is_float64(t):
        return Float64(nullable)
    if pa.types.is_list(t):
        return List(t.value_type, nullable)
    if pa.types.is_struct(t):
        return pandastype_to_dtype(t.to_pandas_dtype(), True)
    if pa.types.is_null(t):
        return Void()
    if pa.types.is_string(t):
        return String(nullable)
    if pa.types.is_map(t):
        return Map(t.item_type, t.key_type, nullable)
    raise NotImplementedError("unsupported case")
