import operator
import unittest
from typing import List, Optional, cast

# Skipping analyzing 'numpy': found module but no type hints or library stubs
import numpy as np  # type: ignore
import numpy.ma as ma  # type: ignore


# Skipping analyzing 'pandas': found module but no type hints or library stubs
import pandas as pd  # type: ignore
import pyarrow as pa  # type: ignore
from torcharrow import (
    Session,
    AbstractColumn,
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
    is_primitive,
    typeof_np_dtype,
    is_boolean_or_numerical
)

import torcharrow as ta


def from_arrow_table(
    table, dtype: Optional[DType] = None, columns: Optional[List[str]] = None
    ,session = None, to = ''
):
    """ "
    Convert arrow table to a torcharrow dataframe.
    """
    session = session or Session.default
    to = to or session.to
    assert isinstance(table, pa.Table)
    if dtype is not None:
        assert is_struct(dtype)
        dtype = cast(Struct, dtype)
        res = {}
        for f in dtype.fields:
            chunked_array = table.column(f.name)
            pydata = chunked_array.to_pylist()
            res[f.name] = session.Column(pydata, f.dtype)
        return session.DataFrame(res, to = to)
    else:
        res = {}
        table = table.select(columns) if columns is not None else table
        for n in table.column_names:
            chunked_array = table.column(n)
            pydata = chunked_array.to_pylist()
            res[n] = session.Column(
                pydata,
                dtype=_arrowtype_to_dtype(
                    table.schema.field(n).type, table.column(n).null_count > 0
                ),
            )
        return session.DataFrame(res, to = to)


def from_pandas_dataframe(
    df, dtype: Optional[DType] = None, columns: Optional[List[str]] = None
    ,session = None, to = ''
):
    """ "
    Convert pandas dataframe to  torcharrow dataframe (drops indices).
    """
    session = session or Session.default
    to = to or session.to

    if dtype is not None:
        assert is_struct(dtype)
        dtype = cast(Struct, dtype)
        res = {}
        for f in dtype.fields:
            # this shows that Column shoud also construct Dataframes!
            res[f.name] = from_pandas_series(pd.Series(df[f.name]))
        return session.Frame(res, dtype = dtype,to = to)
    else:
        res = {}
        for n in df.columns:
            if columns is None or n in columns:
                res[n] = from_pandas_series(pd.Series(df[n]))
        return session.Frame(res, to = to)


def from_arrow_array(array, dtype=None, session = None, to = ''):
    """ "
    Convert arrow array to a torcharrow column.
    """
    session = session or Session.default
    to = to or session.to
    assert isinstance(array, pa.Array)
    pydata = _arrow_scalar_to_py(array)
    if dtype is not None:
        assert not is_struct(dtype)
        return session.Column(pydata, dtype, to=to)
    else:
        return session.Column(
            pydata, dtype=_arrowtype_to_dtype(array.type, array.null_count > 0), to=to
        )


def from_pandas_series(series, dtype=None,session = None, to = ''):
    """ "
    Convert pandas series array to a torcharrow column (drops indices).
    """
    session = session or Session.default
    to = to or session.to

    return from_numpy(series.to_numpy(), dtype, session, to)

def from_numpy(array, dtype, session = None, to= ''):
    """
    Convert 1dim numpy array to a torcharrow column (zero copy).
    """
    session = session or Session.default
    to = to or session.to

    if isinstance(array, ma.core.MaskedArray) and array.ndim==1:
        return _from_numpy_ma(array.data, array.mask, dtype,session,to)
    elif isinstance(array, np.ndarray) and array.ndim==1:
        return _from_numpy_nd(array, dtype,session,to)
    else:
        raise TypeError(f"cannot convert numpy array of type {array.dtype}")

def _is_not_str(s): return not isinstance(s, str)

def _from_numpy_ma(data, mask, dtype, session = None, to= ''):
    # adopt types
    if dtype is None:
        dtype = typeof_nptype(data.dtype).with_null()
    else:
        assert is_primitive_type(dtype)
        assert dtype == typeof_nptype(data.dtype).with_null()
        #TODO if not, adopt the type or?
        # Something like ma.array
        # np.array([np.nan, np.nan,  3.]).astype(np.int64), 
        # mask = np.isnan([np.nan, np.nan,  3.]))

    # create column, only zero copy supported
    if  is_boolean_or_numerical(dtype):
        assert not np.all(np.isnan(ma.array(data,mask).compressed()))

        return session._Full(data, dtype= dtype, mask=mask)
    elif is_string(dtype) or _is_object(dtype): 
        assert np.all(np.vectorize(_is_not_str)(ma.array(data,mask).compressed()))
        return session._Full(data, dtype= dtype, mask=mask)
    else:
        raise TypeError(f"cannot convert masked numpy array of type {data.dtype}")

def _from_numpy_nd(data, dtype, session = None, to= ''):
    # adopt types
    if dtype is None:
        dtype = typeof_np_dtype(data.dtype)
        if dtype is None:
            dtype = string   
    else:
        assert is_primitive(dtype)
        #TODO Check why teh following assert  isn't the case
        #assert dtype == typeof_np_dtype(data.dtype)

    # create column, only zero copy supported
    if  is_boolean_or_numerical(dtype):
        mask = np.isnan(data)
        return session._Full(data, dtype= dtype, mask=mask)
    elif is_string(dtype): 
        mask = np.vectorize(_is_not_str)(data)
        if np.any(mask):
            dtype = dtype.with_null()
        return session._Full(data, dtype= dtype, mask=mask)
    else:
        raise TypeError("can not convert numpy array of type {data.dtype,}")




# def _column_without_nan(series, dtype):
#     if dtype is None or is_floating(dtype):
#         for i in series:
#             if isinstance(i, float) and np.isnan(i):
#                 yield None
#             else:
#                 yield i
#     else:
#         for i in series:
#             yield i


def _arrow_scalar_to_py(array):
    for i in array:
        yield i.as_py()


def _pandatype_to_dtype(t, nullable):
    return typeof_nptype(t, nullable)


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


