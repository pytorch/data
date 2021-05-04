import array as ar
import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Union

import numpy as np
import numpy.ma as ma
from numpy.lib.function_base import append
from torcharrow.column import AbstractColumn
from torcharrow.dtypes import (NL, Boolean, DType, Field, Float32, Float64,
                               Int8, Int16, Int32, Int64, ScalarTypes, Struct,
                               float64, is_boolean, is_boolean_or_numerical,
                               is_numerical, np_typeof_dtype, string,
                               typeof_np_dtype)
from torcharrow.expression import expression
from torcharrow.numerical_column import NumericalColumn
from torcharrow.session import ColumnFactory
from torcharrow.tabulate import tabulate
from torcharrow.trace import trace

# ------------------------------------------------------------------------------


class NumericalColumnCpu(NumericalColumn):
    """A Numerical Column"""

    # NumericalColumnCpu is currently exactly the same code as
    # NumericalColumnTest
    #
    # However it uses the factory key 'cpu',e.g. see
    #
    # ColumnFactory.register(
    #       (dtype.typecode+"_empty", 'cpu'), NumericalColumnCpu._empty)
    # ...

    # Velox can implement whatever it wants to implement,
    # the only contract it has to obey is that
    # - the signatures for all public APIs must stay the same
    # - the signature of the internal builders must stay the same, e.g
    # _full, _empty, _append_null, _append_value, _append_data, _finalize

    # private
    def __init__(self, session, to, dtype, data, mask):
        assert is_boolean_or_numerical(dtype)
        super().__init__(session, to, dtype)
        self._data = data  # Union[ar.array.np.ndarray]
        self._mask = mask   # Union[ar.array.np.ndarray]

    @staticmethod
    def _full(session, to, data, dtype=None, mask=None):
        assert isinstance(data, np.ndarray) and data.ndim == 1
        if dtype is None:
            dtype = typeof_np_ndarray(data.dtype)
        else:
            if dtype != typeof_np_dtype(data.dtype):
                # TODO fix nullability
                # raise TypeError(f'type of data {data.dtype} and given type {dtype} must be the same')
                pass
        if not is_boolean_or_numerical(dtype):
            raise TypeError(
                f'construction of columns of type {dtype} not supported')
        if mask is None:
            mask = AbstractColumn._valid_mask(len(data))
        elif len(data) != len(mask):
            raise ValueError(
                f'data length {len(data)} must be the same as mask length {len(mask)}')
        # TODO check that all non-masked items are legal numbers (i.e not nan)
        return NumericalColumnCpu(session, to, dtype, data, mask)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(session, to, dtype, mask=None):
        _mask = mask if mask is not None else ar.array("b")
        return NumericalColumnCpu(session, to, dtype, ar.array(dtype.arraycode), _mask)

    def _append_null(self):
        self._mask.append(True)
        self._data.append(self.dtype.default)

    def _append_value(self, value):
        self._mask.append(False)
        self._data.append(value)

    def _append_data(self, value):
        self._data.append(value)

    def _finalize(self):
        self._data = np.array(
            self._data, dtype=np_typeof_dtype(self.dtype), copy=False)
        if isinstance(self._mask, (bool, np.bool_)):
            self._mask = np.full((len(self._data),),
                                 self._mask, dtype=np.bool_)
        elif isinstance(self._mask, ar.array):
            self._mask = np.array(self._mask, dtype=np.bool_, copy=False)
        else:
            assert isinstance(self._mask, np.ndarray)
        return self

    # @trace
    def __len__(self):
        return len(self._data)

    @trace
    def null_count(self):
        """Return number of null items"""
        return sum(self._mask) if self.isnullable else 0

    def copy(self):
        return NumericalColumnCpu(*self._meta(), self._data.copy(), self.mask.copy())

    def getdata(self, i):
        return self._data[i]

    def getmask(self, i):
        return self._mask[i]

    @trace
    def gets(self, indices):
        return NumericalColumnCpu(*self._meta(), self._data[indices], self._mask[indices])

    @trace
    def slice(self, start, stop, step):
        range = slice(start, stop, step)
        return NumericalColumnCpu(*self._meta(), self._data[range], self._mask[range])

    # append ----------------------------------------------------------------
    @trace
    def append(self, values):
        tmp = self.session.Column(values, dtype=self.dtype, to=self.to)
        return NumericalColumnCpu(*self._meta(),
                                  np.append(self._data, tmp._data),
                                  np.append(self._mask, tmp._mask))

    # printing ----------------------------------------------------------------

    def __str__(self):
        return f"Column([{', '.join(str(i) for i in self)}], id = {self.id})"

    def __repr__(self):
        tab = tabulate(
            [[l if l is not None else "None"] for l in self],
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self._dtype}, length: {len(self)}, null_count: {self.null_count()}"
        return tab + NL + typ


# sorting -----------------------------------------------------------------


    @ trace
    @ expression
    def sort(
        self,
        columns: Optional[List[str]] = None,
        ascending=True,
        na_position: Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order"""
        if columns is not None:
            raise TypeError(
                "sort on numerical column can't have 'columns' parameter"
            )
        res = ma.array(self._data.copy(), mask=self._mask.copy())
        if self.isnullable:
            res.sort(endwith=(na_position == "last"))
        else:
            res.sort()
        if ascending:
            return NumericalColumnCpu(*self._meta(), res.data, res.mask)
        else:
            res = np.flip(res)
            return NumericalColumnCpu(*self._meta(), res.data, res.mask)

    @ trace
    @ expression
    def nlargest(
        self,
        n=5,
        columns: Optional[List[str]] = None,
        keep: Literal["last", "first"] = "first",
    ):
        """Returns a new data of the *n* largest element."""
        if columns is not None:
            raise TypeError(
                "computing n-largest on numerical column can't have 'columns' parameter"
            )
        return self.sort(columns=None, ascending=False, na_position=keep).head(n)

    @ trace
    @ expression
    def nsmallest(
        self, n=5, columns: Optional[List[str]] = None, keep="first"
    ):
        """Returns a new data of the *n* smallest element. """
        if columns is not None:
            raise TypeError(
                "computing n-smallest on numerical column can't have 'columns' parameter"
            )

        return self.sort(columns=None, ascending=True, na_position=keep).head(n)

    @ trace
    @ expression
    def nunique(self, dropna=True):
        """Returns the number of unique values of the column"""
        if dropna:
            return len(np.unique(self._ma().compressed()))
        else:
            return len(np.unique(self._ma()))

    # operators ---------------------------------------------------------------

    def _ma(self):
        return ma.array(self._data, mask=self._mask)

    def check_same(self, other):
        if self.session != other.session:
            raise TypeError('can only have one session per app.')
        if self.to != other.to:
            raise TypeError('self and other must have same device.')

    def _from_ma(self, masked_array):
        dtype = _numpytype_to_dtype(
            masked_array.dtype, ma.is_masked(masked_array))
        mask = masked_array.mask if not isinstance(
            masked_array.mask, np.bool_) else np.full((len(masked_array),), masked_array.mask)
        return NumericalColumnCpu(self.session, self.to, dtype, masked_array.data, mask)

    @trace
    @ expression
    def __add__(self, other):
        if isinstance(other, NumericalColumn):
            self.check_same(other)
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma()+other._ma())
        else:
            return self._from_ma(self._ma()+other)

    @trace
    @ expression
    def __radd__(self, other):
        print("RCPU", self, other)
        if isinstance(other, NumericalColumn):
            self.session.check_is_same(other.session)
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma()+self._ma())
        else:
            return self._from_ma(other + self._ma())

    @trace
    @ expression
    def __sub__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma()-other._ma())
        else:
            return self._from_ma(self._ma()-other)

    @trace
    @ expression
    def __rsub__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma()-self._ma())
        else:
            return self._from_ma(other - self._ma())

    @trace
    @ expression
    def __mul__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma()*other._ma())
        else:
            return self._from_ma(self._ma()*other)

    @trace
    @ expression
    def __rmul__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma()*self._ma())
        else:
            return self._from_ma(other * self._ma())

    @trace
    @ expression
    def __floordiv__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma()//other._ma())
        else:
            return self._from_ma(self._ma()//other)

    @trace
    @ expression
    def __rfloordiv__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma()//self._ma())
        else:
            return self._from_ma(other // self._ma())

    @trace
    @ expression
    def __truediv__(self, other):
        # forces FloatingPointError: divide by zero encountered in true_divide
        np.seterr(divide='raise')
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma()/other._ma())
        else:
            return self._from_ma(self._ma()/other)

    @trace
    @ expression
    def __rtruediv__(self, other):
        # forces FloatingPointError: divide by zero encountered in true_divide
        np.seterr(divide='raise')
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma()/self._ma())
        else:
            return self._from_ma(other / self._ma())

    @trace
    @ expression
    def __mod__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() % other._ma())
        else:
            return self._from_ma(self._ma() % other)

    @trace
    @ expression
    def __rmod__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma()+self._ma())
        else:
            return self._from_ma(other + self._ma())

    @trace
    @ expression
    def __pow__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma()**other._ma())
        else:
            return self._from_ma(self._ma()**other)

    @trace
    @ expression
    def __rpow__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma()**self._ma())
        else:
            return self._from_ma(other ** self._ma())

    @trace
    @ expression
    def __eq__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() == other._ma())
        else:
            return self._from_ma(self._ma() == other)

    @trace
    @ expression
    def __ne__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() != other._ma())
        else:
            return self._from_ma(self._ma() != other)

    @trace
    @ expression
    def __lt__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() < other._ma())
        else:
            return self._from_ma(self._ma() < other)

    @trace
    @ expression
    def __gt__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() > other._ma())
        else:
            return self._from_ma(self._ma() > other)

    @trace
    @ expression
    def __le__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() <= other._ma())
        else:
            return self._from_ma(self._ma() <= other)

    @trace
    @ expression
    def __ge__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() > other._ma())
        else:
            return self._from_ma(self._ma() > other)
    # bitwise or (|), used for logical or

    @trace
    @ expression
    def __or__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() | other._ma())
        else:
            return self._from_ma(self._ma() | other)

    @trace
    @ expression
    def __ror__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma() | self._ma())
        else:
            return self._from_ma(other | self._ma())

    # bitwise and (&), used for logical and

    @trace
    @ expression
    def __and__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() & other._ma())
        else:
            return self._from_ma(self._ma() & other)

    @trace
    @ expression
    def __rand__(self, other):
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma() & self._ma())
        else:
            return self._from_ma(other & self._ma())

    # bitwise complement (~), used for logical not

    @trace
    @ expression
    def __invert__(self):
        return self._from_ma(np.logical_not(self._ma()))

    # unary arithmetic

    @trace
    @ expression
    def __neg__(self):
        return self._from_ma(-(self._ma()))

    @trace
    @ expression
    def __pos__(self):
        return self._from_ma(+(self._ma()))

    # unary math related

    @trace
    @ expression
    def abs(self):
        """Absolute value of each element of the series."""
        return self._from_ma(np.abs(self._ma()))

    @trace
    @ expression
    def ceil(self):
        """Rounds each value upward to the smallest integral"""
        return self._from_ma(np.ceil(self._ma()))

    @trace
    @ expression
    def floor(self):
        """Rounds each value downward to the largest integral value"""
        return self._from_ma(np.floor(self._ma()))

    @trace
    @ expression
    def round(self, decimals=0):
        """Round each value in a data to the given number of decimals."""
        return self._from_ma(np.round(self._ma(), decimals))

    # @expression
    # def hash_values(self):
    #     """Compute the hash of values in this column."""
    #     return self._unary_operator(hash, Int64(self.dtype.nullable))

    # isin --------------------------------------------------------------------
    @ trace
    @ expression
    def isin(self, values, invert=False):
        """Check whether list values are contained in data, or column/dataframe (row/column specific)."""
        # Todo decide on wether mask matters?
        data = np.isin(self._data, values, invert=invert)
        false_ = np.full_like(self._data, False)
        return self._from_ma(ma.array(np.where(self._mask, ~self._mask, data), mask=false_))

    # data cleaning -----------------------------------------------------------

    @ trace
    @ expression
    def fillna(
        self, fill_value: Union[ScalarTypes, Dict, AbstractColumn, Literal[None]]
    ):
        if not isinstance(fill_value, (int, float, bool)):
            raise TypeError(f"fillna with {type(fill_value)} is not supported")
        if not self.isnullable:
            return self
        else:
            full = np.full_like(self._data, fill_value)
            false_ = np.full_like(self._data, False)
            return self._from_ma(ma.array(np.where(self._mask, full, self._data), mask=false_))

    @ trace
    @ expression
    def dropna(self, how: Literal["any", "all"] = "any"):
        """Return a column with rows removed where a row has any or all nulls."""
        if not self.isnullable:
            return self
        else:
            dropped = ma.array(self._data, mask=self._mask).compressed()
            false_ = np.full_like(dropped, False)
            return self._from_ma(ma.array(dropped, mask=false_))

    @ trace
    @ expression
    def drop_duplicates(
        self,
        subset: Optional[List[str]] = None,
    ):
        """ Remove duplicate values from row/frame """
        if subset is not None:
            raise TypeError(
                f"subset parameter for numerical columns not supported")
        return self._from_ma(np.unique(ma.array(self._data, mask=self._mask)))

    # universal  ---------------------------------------------------------------

    staticmethod

    def _raise(error):
        raise error

    @ trace
    @ expression
    # def iif(self, then_, else_):
    #     """Equivalent to ternary expression: if self then then_ else else_"""
    #     # FIXME this assue staht everyone has a _np field...
    #     then_np = then_._val if hasattr(
    #         then_, "_np") else np.full_like(self._ma(), then_)
    #     else_np = else_._val if hasattr(
    #         else_, "_np") else np.full_like(self._ma(), else_)
    #     return self._from_ma(then_.where(self._ma(), then_np, else_np))
    @ trace
    @ expression
    def min(self, numeric_only=None, fill_value=None):
        """Return the minimum of the non-null values of the Column."""
        m = np.ma.min(self._ma(), fill_value)
        return m if m is not ma.masked else NumericalColumnCpu.raise_(ValueError(f'min returns {m}'))

    @ trace
    @ expression
    def max(self, numeric_only=None, fill_value=None):
        """Return the maximum of the non-null values of the column."""
        m = np.ma.max(self._ma(), fill_value)
        return m if m is not ma.masked else NumericalColumnCpu.raise_(ValueError(f'max returns {m}'))

    @ trace
    @ expression
    def all(self, boolean_only=None):
        """Return whether all non-null elements are True in Column"""
        m = np.ma.all(self._ma())
        return bool(m) if m is not ma.masked else NumericalColumnCpu.raise_(ValueError(f'all returns {m}'))

    @ trace
    @ expression
    def any(self, skipna=True, boolean_only=None):
        """Return whether any non-null element is True in Column"""
        m = np.ma.any(self._ma())
        return bool(m) if m is not ma.masked else NumericalColumnCpu.raise_(ValueError(f'any returns {m}'))

    @ trace
    @ expression
    def sum(self):
        # TODO Should be def sum(self, initial=None) but didn't get to work
        """Return sum of all non-null elements in Column (starting with initial)"""
        m = np.ma.sum(self._ma())
        return m if m is not ma.masked else NumericalColumnCpu.raise_(ValueError(f'sum returns {m}'))

    @ trace
    @ expression
    def prod(self):
        """Return produce of the values in the data"""
        m = np.ma.prod(self._ma())
        return m if m is not ma.masked else NumericalColumnCpu.raise_(ValueError(f'prod returns {m}'))

    @ trace
    @ expression
    def cummin(self):
        """Return cumulative minimum of the data."""
        if not self.isnullable:
            return self._from_ma(np.minimum.accumulate(self._ma()))
        else:
            mx = self._ma().max()
            cs = np.where(self._ma().mask, mx+1, self._ma())
            rs = ma.array(np.minimum.accumulate(cs), mask=self._ma().mask)
            return self._from_ma(rs)

    @ trace
    @ expression
    def cummax(self):
        """Return cumulative maximum of the data."""
        if not self.isnullable:
            return self._from_ma(np.maximum.accumulate(self._ma()))
        else:
            mx = self._ma().min()
            cs = np.where(self._ma().mask, mx-1, self._ma())
            rs = ma.array(np.maximum.accumulate(cs), mask=self._ma().mask)
            return self._from_ma(rs)

    @ trace
    @ expression
    def cumsum(self):
        """Return cumulative sum of the data."""
        return self._from_ma(np.cumsum(self._ma()))

    @ trace
    @ expression
    def cumprod(self):
        """Return cumulative product of the data."""
        return self._from_ma(np.cumprod(self._ma()))

    @ trace
    @ expression
    def mean(self):
        """Return the mean of the values in the series."""
        return np.ma.mean(self._ma())

    @ trace
    @ expression
    def median(self):
        """Return the median of the values in the data."""
        return np.ma.median(self._ma())

    # @ trace
    # @ expression
    # def mode(self):
    #     """Return the mode(s) of the data."""
    #     return np.ma.mode(self._ma())

    @ trace
    @ expression
    def std(self, ddof=1):
        """Return the stddev(s) of the data."""
        return np.ma.std(self._ma(),  ddof=ddof)

    @ trace
    @ expression
    def percentiles(self, q, interpolation='midpoint'):
        """ Compute the q-th percentile of non-null data."""
        # If q is a single percentile, then the result is a scalar.
        # If multiple percentiles are given, the result corresponds to the percentiles.
        return list(np.percentile(self._ma().compressed(), q, interpolation=interpolation))

    # describe ----------------------------------------------------------------
    @ trace
    @ expression
    def describe(
        self,
        percentiles_=None,
        include_columns: Union[List[DType], Literal[None]] = None,
        exclude_columns: Union[List[DType], Literal[None]] = None,
    ):
        """Generate descriptive statistics."""

        # Not supported: datetime_is_numeric=False,
        if include_columns is not None or exclude_columns is not None:
            raise TypeError(
                f"'include/exclude columns' parameter for '{type(self).__name__}' not supported "
            )
        if percentiles_ is None:
            percentiles_ = [25, 50, 75]
        percentiles_ = sorted(set(percentiles_))
        if len(percentiles_) > 0:
            if percentiles_[0] < 0 or percentiles_[-1] > 100:
                raise ValueError("percentiles must be betwen 0 and 100")

        if is_numerical(self.dtype):
            res = self.session._Empty(
                Struct([Field("statistic", string), Field("value", float64)]), to=self.to
            )
            res._append(("count", self.count()))
            res._append(("mean", self.mean()))
            res._append(("std", self.std()))
            res._append(("min", self.min()))
            values = self.percentiles(percentiles_, "midpoint")
            for p, v in zip(percentiles_, values):
                res._append((f"{p}%", v))
            res._append(("max", self.max()))
            return res._finalize()
        else:
            raise ValueError(f"median undefined for {type(self).__name__}.")

    # Flat column specific ops ----------------------------------------------------------

    @ trace
    @ expression
    def is_unique(self):
        """Return boolean if data values are unique."""
        return len(self._ma()) == len(np.unique(self._ma()))

    @ trace
    @ expression
    def is_monotonic_increasing(self):
        """Return boolean if values in the object are monotonic increasing"""
        return np.all(self._ma()[:-1] < self._ma()[1:])

    @ trace
    @ expression
    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonic decreasing"""
        return np.all(self._ma()[:-1] < self._ma()[1:])

  # conversions -------------------------------------------------------------

    @ trace
    def astype(self, dtype):
        """Cast the Column to the given dtype"""
        return self._from_ma(self._ma().astype(np_typeof(dtype)))

    # interop ----------------------------------------------------------------

    @ trace
    def to_pandas(self):
        """Convert self to pandas dataframe"""
        # TODO Add type translation
        # Skipping analyzing 'pandas': found module but no type hints or library stubs
        import pandas as pd  # type: ignore

        return pd.Series(self._ma())

    @ trace
    def to_arrow(self):
        """Convert self to pandas dataframe"""
        # TODO Add type translation
        import pyarrow as pa  # type: ignore

        return pa.array(self._ma())


# ------------------------------------------------------------------------------
# registering all numeric and boolean types for the factory...
for dtype in {Int8, Int16, Int32, Int64, Float32, Float64, Boolean}:
    ColumnFactory.register(
        (dtype.typecode+"_empty", 'cpu'), NumericalColumnCpu._empty)

# registering all numeric and boolean types for the factory...
for dtype in {Int8, Int16, Int32, Int64, Float32, Float64, Boolean}:
    ColumnFactory.register(
        (dtype.typecode+"_full", 'cpu'), NumericalColumnCpu._full)


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
