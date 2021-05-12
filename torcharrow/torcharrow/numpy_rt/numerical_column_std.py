import array as ar
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import numpy.ma as ma

import torcharrow.dtypes as dt
from torcharrow.icolumn import IColumn
from torcharrow.column_factory import ColumnFactory
from torcharrow.expression import expression
from torcharrow.inumerical_column import INumericalColumn
from torcharrow.trace import trace

# ------------------------------------------------------------------------------


class NumericalColumnStd(INumericalColumn):
    """A Numerical Column implemented by Numpy"""

    # implements all operations from column --
    # unless the column operations is completely generic
    # and its perf can't be improved upon

    def __init__(self, scope, to, dtype, data, mask):
        # private constructor
        #   DON'T CALL: NumericalColumnStd
        assert dt.is_boolean_or_numerical(dtype)
        super().__init__(scope, to, dtype)
        self._data = data  # Union[ar.array.np.ndarray]
        self._mask = mask  # Union[ar.array.np.ndarray]

    # private factory and builders --------------------------------------------

    @staticmethod
    def _full(scope, to, data, dtype=None, mask=None):
        # DON'T CALL: _full USE _FullColumn instead

        assert isinstance(data, np.ndarray) and data.ndim == 1
        if dtype is None:
            dtype = dt.typeof_np_ndarray(data.dtype)
        else:
            if dtype != dt.typeof_np_dtype(data.dtype):
                # TODO fix nullability
                # raise TypeError(f'type of data {data.dtype} and given type {dtype} must be the same')
                pass
        if not dt.is_boolean_or_numerical(dtype):
            raise TypeError(f"construction of columns of type {dtype} not supported")
        if mask is None:
            mask = NumericalColumnStd._valid_mask(len(data))
        elif len(data) != len(mask):
            raise ValueError(
                f"data length {len(data)} must be the same as mask length {len(mask)}"
            )
        # TODO check that all non-masked items are legal numbers (i.e not nan)
        return NumericalColumnStd(scope, to, dtype, data, mask)

    @staticmethod
    def _empty(scope, to, dtype, mask=None):
        # DON'T CALL: _empty USE _EmptyColumn instead
        # Any _empty must be followed by a _finalize; no other ops are allowed during this time
        _mask = mask if mask is not None else ar.array("b")
        return NumericalColumnStd(scope, to, dtype, ar.array(dtype.arraycode), _mask)

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
            self._data, dtype=dt.np_typeof_dtype(self.dtype), copy=False
        )
        if isinstance(self._mask, (bool, np.bool8)):
            self._mask = np.full((len(self._data),), self._mask, dtype=np.bool8)
        elif isinstance(self._mask, ar.array):
            self._mask = np.array(self._mask, dtype=np.bool8, copy=False)
        else:
            assert isinstance(self._mask, np.ndarray)
        return self

    @staticmethod
    def _valid_mask(ct):
        return np.full((ct,), False, dtype=np.bool8)

    # observers selectors and getters ---------------------------------------------------

    def __len__(self):
        return len(self._data)

    def null_count(self):
        """Return number of null items"""
        return sum(self._mask) if self.isnullable else 0

    def getdata(self, i):
        return self._data[i]

    def getmask(self, i):
        return self._mask[i]

    @trace
    def gets(self, indices):
        return self.scope._FullColumn(
            self._data[indices], self.dtype, self.to, self._mask[indices]
        )

    @trace
    def slice(self, start, stop, step):
        range = slice(start, stop, step)
        return self.scope._FullColumn(
            self._data[range], self.dtype, self.to, self._mask[range]
        )

    # public append/copy/astype------------------------------------------------

    @trace
    def append(self, values):
        tmp = self.scope.Column(values, dtype=self.dtype, to=self.to)
        return self.scope._FullColumn(
            np.append(self._data, tmp._data),
            self.dtype,
            self.to,
            np.append(self._mask, tmp._mask),
        )

    @trace
    def copy(self):
        return self.scope._FullColumn(
            self._data.copy(), self.dtype, self.to, self.mask.copy()
        )

    def astype(self, dtype):
        """Cast the Column to the given dtype"""
        if dt.is_is_primitive(dtype):
            return self._from_ma(
                self._ma().astype(dt.np_typeof_dtype(dtype, casting="same_kind"))
            )
        else:
            raise TypeError('f"{astype} for {type(self).__name__} is not supported")')

    # if-then-else ---------------------------------------------------------------

    def ite(self, then_, else_):
        """Vectorized if-then-else"""
        if not dt.is_boolean(self.dtype):
            raise TypeError("condition must be a boolean vector")
        if not isinstance(then_, IColumn):
            then_ = self._Column(then_)
        if not isinstance(else_, IColumn):
            else_ = self._Column(else_)
        lub = dt.common_dtype(then_.dtype, else_.dtype)

        if lub is None or dt.is_void(lub):
            raise TypeError(
                "then and else branches must have compatible types, got {then_.dtype} and {else_.dtype}, respectively"
            )
        if isinstance(then_, NumericalColumnStd) and isinstance(
            else_, NumericalColumnStd
        ):

            return self._from_ma(np.ma.where(self._ma(), then_._ma(), else_._ma()))
        else:
            # refer back to default handling...
            return super.ite(self, then_, else_)

    # sorting, top-k, unique---------------------------------------------------

    @trace
    @expression
    def sort(
        self,
        columns: Optional[List[str]] = None,
        ascending=True,
        na_position: Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order"""
        if columns is not None:
            raise TypeError("sort on numerical column can't have 'columns' parameter")
        res = ma.array(self._data.copy(), mask=self._mask.copy())
        if self.isnullable:
            res.sort(endwith=(na_position == "last"))
        else:
            res.sort()
        if ascending:
            return self.scope._FullColumn(res.data, self.dtype, self.to, res.mask)
        else:
            res = np.flip(res)
            return self.scope._FullColumn(res.data, self.dtype, self.to, res.mask)

    @trace
    @expression
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

    @trace
    @expression
    def nsmallest(self, n=5, columns: Optional[List[str]] = None, keep="first"):
        """Returns a new data of the *n* smallest element."""
        if columns is not None:
            raise TypeError(
                "computing n-smallest on numerical column can't have 'columns' parameter"
            )

        return self.sort(columns=None, ascending=True, na_position=keep).head(n)

    @trace
    @expression
    def nunique(self, dropna=True):
        """Returns the number of unique values of the column"""
        if dropna:
            return len(np.unique(self._ma().compressed()))
        else:
            return len(np.unique(self._ma()))

    # operators ---------------------------------------------------------------

    def _ma(self):
        return ma.array(self._data, mask=self._mask)

    def _from_ma(self, masked_array):
        dtype = dt.typeof_np_dtype(masked_array.dtype).with_null(
            ma.is_masked(masked_array)
        )
        mask = (
            masked_array.mask
            if not isinstance(masked_array.mask, np.bool8)
            else np.full((len(masked_array),), masked_array.mask)
        )
        return NumericalColumnStd(self.scope, self.to, dtype, masked_array.data, mask)

    @trace
    @expression
    def __add__(self, other):
        """Vectorized a + b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() + other._ma())
        else:
            return self._from_ma(self._ma() + other)

    @trace
    @expression
    def __radd__(self, other):
        """Vectorized b + a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            self.scope.check_is_same(other.scope)
            return self._from_ma(other._ma() + self._ma())
        else:
            return self._from_ma(other + self._ma())

    @trace
    @expression
    def __sub__(self, other):
        """Vectorized a - b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() - other._ma())
        else:
            return self._from_ma(self._ma() - other)

    @trace
    @expression
    def __rsub__(self, other):
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(other._ma() - self._ma())
        else:
            return self._from_ma(other - self._ma())

    @trace
    @expression
    def __mul__(self, other):
        """Vectorized a * b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() * other._ma())
        else:
            return self._from_ma(self._ma() * other)

    @trace
    @expression
    def __rmul__(self, other):
        """Vectorized b * a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(other._ma() * self._ma())
        else:
            return self._from_ma(other * self._ma())

    @trace
    @expression
    def __floordiv__(self, other):
        """Vectorized a // b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() // other._ma())
        else:
            return self._from_ma(self._ma() // other)

    @trace
    @expression
    def __rfloordiv__(self, other):
        """Vectorized b // a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(other._ma() // self._ma())
        else:
            return self._from_ma(other // self._ma())

    @trace
    @expression
    def __truediv__(self, other):
        """Vectorized a / b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        # forces FloatingPointError: divide by zero encountered in true_divide
        np.seterr(divide="raise")
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() / other._ma())
        else:
            return self._from_ma(self._ma() / other)

    @trace
    @expression
    def __rtruediv__(self, other):
        """Vectorized b / a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        # forces FloatingPointError: divide by zero encountered in true_divide
        np.seterr(divide="raise")
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(other._ma() / self._ma())
        else:
            return self._from_ma(other / self._ma())

    @trace
    @expression
    def __mod__(self, other):
        """Vectorized a % b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() % other._ma())
        else:
            return self._from_ma(self._ma() % other)

    @trace
    @expression
    def __rmod__(self, other):
        """Vectorized b % a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(other._ma() + self._ma())
        else:
            return self._from_ma(other + self._ma())

    @trace
    @expression
    def __pow__(self, other):
        """Vectorized a ** b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() ** other._ma())
        else:
            return self._from_ma(self._ma() ** other)

    @trace
    @expression
    def __rpow__(self, other):
        """Vectorized b ** a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(other._ma() ** self._ma())
        else:
            return self._from_ma(other ** self._ma())

    @trace
    @expression
    def __eq__(self, other):
        """Vectorized a == b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() == other._ma())
        else:
            return self._from_ma(self._ma() == other)

    @trace
    @expression
    def __ne__(self, other):
        """Vectorized a != b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() != other._ma())
        else:
            return self._from_ma(self._ma() != other)

    @trace
    @expression
    def __lt__(self, other):
        """Vectorized a < b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() < other._ma())
        else:
            return self._from_ma(self._ma() < other)

    @trace
    @expression
    def __gt__(self, other):
        """Vectorized a > b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() > other._ma())
        else:
            return self._from_ma(self._ma() > other)

    @trace
    @expression
    def __le__(self, other):
        """Vectorized a < b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() <= other._ma())
        else:
            return self._from_ma(self._ma() <= other)

    @trace
    @expression
    def __ge__(self, other):
        """Vectorized a <= b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() > other._ma())
        else:
            return self._from_ma(self._ma() > other)

    @trace
    @expression
    def __or__(self, other):
        """Vectorized boolean or: a | b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() | other._ma())
        else:
            return self._from_ma(self._ma() | other)

    @trace
    @expression
    def __ror__(self, other):
        """Vectorized boolean reverse or: b | a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(other._ma() | self._ma())
        else:
            return self._from_ma(other | self._ma())

    @trace
    @expression
    def __and__(self, other):
        """Vectorized boolean and: a & b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(self._ma() & other._ma())
        else:
            return self._from_ma(self._ma() & other)

    @trace
    @expression
    def __rand__(self, other):
        """Vectorized boolean reverse and: b & a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnStd):
            return self._from_ma(other._ma() & self._ma())
        else:
            return self._from_ma(other & self._ma())

    @trace
    @expression
    def __invert__(self):
        """Vectorized boolean not: ~ a."""
        return self._from_ma(np.logical_not(self._ma()))

    @trace
    @expression
    def __neg__(self):
        """Vectorized: - a."""
        return self._from_ma(-(self._ma()))

    @trace
    @expression
    def __pos__(self):
        """Vectorized: + a."""
        return self._from_ma(+(self._ma()))

    @trace
    @expression
    def isin(self, values, invert=False):
        """Check whether list values are contained in data, or column/dataframe (row/column specific)."""
        # Todo decide on wether mask matters?
        data = np.isin(self._data, values, invert=invert)
        false_ = np.full_like(self._data, False)
        return self._from_ma(
            ma.array(np.where(self._mask, ~self._mask, data), mask=false_)
        )

    @trace
    @expression
    def abs(self):
        """Absolute value of each element of the series."""
        return self._from_ma(np.abs(self._ma()))

    @trace
    @expression
    def ceil(self):
        """Rounds each value upward to the smallest integral"""
        return self._from_ma(np.ceil(self._ma()))

    @trace
    @expression
    def floor(self):
        """Rounds each value downward to the largest integral value"""
        return self._from_ma(np.floor(self._ma()))

    @trace
    @expression
    def round(self, decimals=0):
        """Round each value in a data to the given number of decimals."""
        return self._from_ma(np.round(self._ma(), decimals))

    # data cleaning -----------------------------------------------------------

    @trace
    @expression
    def fillna(self, fill_value: Union[dt.ScalarTypes, Dict]):
        """Fill NA/NaN values using the specified method."""
        if not isinstance(fill_value, IColumn.scalar_types):
            raise TypeError(f"fillna with {type(fill_value)} is not supported")
        if not self.isnullable:
            return self
        else:
            full = np.full_like(self._data, fill_value)
            false_ = np.full_like(self._data, False)
            return self._from_ma(
                ma.array(np.where(self._mask, full, self._data), mask=false_)
            )

    @trace
    @expression
    def dropna(self, how: Literal["any", "all"] = "any"):
        """Return a column with rows removed where a row has any or all nulls."""
        if not self.isnullable:
            return self
        else:
            dropped = ma.array(self._data, mask=self._mask).compressed()
            false_ = np.full_like(dropped, False)
            return self._from_ma(ma.array(dropped, mask=false_))

    @trace
    @expression
    def drop_duplicates(
        self, subset: Optional[List[str]] = None,
    ):
        """Remove duplicate values from row/frame"""
        if subset is not None:
            raise TypeError(f"subset parameter for numerical columns not supported")
        return self._from_ma(np.unique(ma.array(self._data, mask=self._mask)))

    # universal  ---------------------------------------------------------------

    @trace
    @expression
    def min(self, numeric_only=None, fill_value=None):
        """Return the minimum of the non-null values of the Column."""
        m = np.ma.min(self._ma(), fill_value)
        return (
            m
            if m is not ma.masked
            else NumericalColumnStd.raise_(ValueError(f"min returns {m}"))
        )

    @trace
    @expression
    def max(self, fill_value=None):
        """Return the maximum of the non-null values of the column."""
        m = np.ma.max(self._ma(), fill_value)
        return (
            m
            if m is not ma.masked
            else NumericalColumnStd.raise_(ValueError(f"max returns {m}"))
        )

    @trace
    @expression
    def all(self):
        """Return whether all non-null elements are True in Column"""
        m = np.ma.all(self._ma())
        return (
            bool(m)
            if m is not ma.masked
            else NumericalColumnStd.raise_(ValueError(f"all returns {m}"))
        )

    @trace
    @expression
    def any(self, skipna=True, boolean_only=None):
        """Return whether any non-null element is True in Column"""
        m = np.ma.any(self._ma())
        return (
            bool(m)
            if m is not ma.masked
            else NumericalColumnStd.raise_(ValueError(f"any returns {m}"))
        )

    @trace
    @expression
    def sum(self):
        # TODO Should be def sum(self, initial=None) but didn't get to work
        """Return sum of all non-null elements in Column (starting with initial)"""
        m = np.ma.sum(self._ma())
        return (
            m
            if m is not ma.masked
            else NumericalColumnStd.raise_(ValueError(f"sum returns {m}"))
        )

    @trace
    @expression
    def prod(self):
        """Return produce of the values in the data"""
        m = np.ma.prod(self._ma())
        return (
            m
            if m is not ma.masked
            else NumericalColumnStd.raise_(ValueError(f"prod returns {m}"))
        )

    @trace
    @expression
    def cummin(self):
        """Return cumulative minimum of the data."""
        if not self.isnullable:
            return self._from_ma(np.minimum.accumulate(self._ma()))
        else:
            mx = self._ma().max()
            cs = np.where(self._ma().mask, mx + 1, self._ma())
            rs = ma.array(np.minimum.accumulate(cs), mask=self._ma().mask)
            return self._from_ma(rs)

    @trace
    @expression
    def cummax(self):
        """Return cumulative maximum of the data."""
        if not self.isnullable:
            return self._from_ma(np.maximum.accumulate(self._ma()))
        else:
            mx = self._ma().min()
            cs = np.where(self._ma().mask, mx - 1, self._ma())
            rs = ma.array(np.maximum.accumulate(cs), mask=self._ma().mask)
            return self._from_ma(rs)

    @trace
    @expression
    def cumsum(self):
        """Return cumulative sum of the data."""
        return self._from_ma(np.cumsum(self._ma()))

    @trace
    @expression
    def cumprod(self):
        """Return cumulative product of the data."""
        return self._from_ma(np.cumprod(self._ma()))

    @trace
    @expression
    def mean(self):
        """Return the mean of the values in the series."""
        return np.ma.mean(self._ma())

    @trace
    @expression
    def median(self):
        """Return the median of the values in the data."""
        return np.ma.median(self._ma())

    # @ trace
    # @ expression
    # def mode(self):
    #     """Return the mode(s) of the data."""
    #     return np.ma.mode(self._ma())

    @trace
    @expression
    def std(self, ddof=1):
        """Return the stddev(s) of the data."""
        # ignores nulls
        return np.ma.std(self._ma(), ddof=ddof)

    @trace
    @expression
    def percentiles(self, q, interpolation="midpoint"):
        """Compute the q-th percentile of non-null data."""
        return list(
            np.percentile(self._ma().compressed(), q, interpolation=interpolation)
        )

    # unique and montonic  ----------------------------------------------------

    @trace
    @expression
    def is_unique(self):
        """Return boolean if data values are unique."""
        return len(self._ma()) == len(np.unique(self._ma()))

    @trace
    @expression
    def is_monotonic_increasing(self):
        """Return boolean if values in the object are monotonic increasing"""
        return np.all(self._ma()[:-1] < self._ma()[1:])

    @trace
    @expression
    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonic decreasing"""
        return np.all(self._ma()[:-1] < self._ma()[1:])

    # interop ----------------------------------------------------------------

    @trace
    def to_pandas(self):
        """Convert self to pandas dataframe"""
        # TODO Add type translation
        # Skipping analyzing 'pandas': found module but no type hints or library stubs
        import pandas as pd  # type: ignore

        return pd.Series(self._ma())

    @trace
    def to_arrow(self):
        """Convert self to pandas dataframe"""
        # TODO Add type translation
        import pyarrow as pa  # type: ignore

        return pa.array(self._ma())


# ------------------------------------------------------------------------------
# registering all numeric and boolean types for the factory...
for dtype in {
    dt.Int8,
    dt.Int16,
    dt.Int32,
    dt.Int64,
    dt.Float32,
    dt.Float64,
    dt.Boolean,
}:
    ColumnFactory.register(
        (dtype.typecode + "_empty", "std"), NumericalColumnStd._empty
    )

# registering all numeric and boolean types for the factory...
for dtype in {
    dt.Int8,
    dt.Int16,
    dt.Int32,
    dt.Int64,
    dt.Float32,
    dt.Float64,
    dt.Boolean,
}:
    ColumnFactory.register((dtype.typecode + "_full", "std"), NumericalColumnStd._full)
