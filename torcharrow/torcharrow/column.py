#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from collections import defaultdict
import array as ar
import functools
import math
import operator
import statistics
from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
    Iterable,
)

from .dtypes import (
    DType,
    Field,
    Float64,
    Int64,
    ScalarTypes,
    ScalarTypeValues,
    Struct,
    boolean,
    derive_dtype,
    derive_operator,
    float64,
    infer_dtype_from_prefix,
    is_boolean,
    is_numerical,
    is_primitive,
    is_struct,
    is_tuple,
    string,
)

from .expression import expression
from .trace import trace, traceproperty
from .session import Session


# ------------------------------------------------------------------------------
# Column Factory with default session and device

def Column(
    data: Union[Iterable, DType, Literal[None]] = None,
    dtype: Optional[DType] = None,
    session: Optional[Session] = None,
    to: Device = ''
):
    """
    Column Factory method
    """
    session = session or Session.default
    to = to or session.to
    return session.Column(data, dtype=dtype, to=to)

# ------------------------------------------------------------------------------
# AbstractColumn


class AbstractColumn(Sized, Iterable):
    """AbstractColumn are n vectors (n>=1) of columns"""

    def __init__(self, session, to, dtype):

        self._session = session
        self._to = to
        self._dtype = dtype

        # id handling, used for tracing...
        self.id = f"c{session.ct.next()}"

    # getters ---------------------------------------------------------------

    @property
    def session(self):
        return self._session

    @property
    def to(self):
        return self._to

    @property  # type: ignore
    @traceproperty
    def dtype(self):
        """dtype of the colum/frame"""
        return self._dtype

    @property  # type: ignore
    @traceproperty
    def isnullable(self):
        """A boolean indicating whether column/frame can have nulls"""
        return self.dtype.nullable

    def _meta(self):
        return (self.session, self.to, self.dtype, )

    def _context(self):
        return (self.session, self.to,)

    # private builders -------------------------------------------------------

    def _Empty(self, dtype, mask=None):
        return self.session._Empty(dtype, self.to, mask)

    def _Full(self, data, dtype=None, mask=None):
        return self.session._Full(data, dtype, self.to, mask)

    @trace
    def _append_null(self):
        raise self._not_supported('_append_null')

    @trace
    def _append_value(self, value):
        raise self._not_supported('_append_value')

    @trace
    def _append(self, value):
        if value is None:
            self._append_null()
        else:
            self._append_value(value)

    def _extend(self, values):
        for value in values:
            self._append(value)

    # public builders -------------------------------------------------------

    @trace
    def append(self, values):
        """Returns column/dataframe with values appended."""
        # TODO use _column_copy, but for now this works...
        res = self._Empty(self.dtype)
        for (m, d) in self.items():
            if m:
                res._append_null()
            else:
                res._append_value(d)
        for i in values:
            res._append(i)
        return res._finalize()

    def concat(self, others: List["AbstractColumn"]):
        raise self._not_supported('concat')

    # public simple observers -------------------------------------------------

    @trace
    @expression
    def count(self):
        """Return number of non-NA/null observations pgf the column/frame"""
        return len(self) - self.null_count()

    @trace
    @expression
    def null_count(self):
        """Number of null values"""
        raise self._not_supported('getmask')

    @trace
    @expression
    # @abstractmethod
    def __len__(self):
        """Return number of rows including null values"""
        raise self._not_supported('__len__')

    @trace
    @expression
    # @abstractmethod
    def length(self):
        """Return number of rows including null values"""
        return len(self)

    # private helpers ---------------------------------------------------------

    def _not_supported(self, name):
        raise TypeError(
            f"{name} for type {type(self).__name__} is not supported")

    # selectors ---------------------------------------------------------------
    # @abstractmethod

    def getmask(self, i):
        raise self._not_supported('getmask')

    def getdata(self, i):
        raise self._not_supported('getdata')

    def valid(self, index):
        return not self.getmask(index)

    def get(self, index, fill_value=None):

        if self.getmask(index):
            return fill_value
        else:
            return self.getdata(index)

    @trace
    @expression
    def __getitem__(self, arg):
        """
        If *arg* is a ``str`` type, return the column named *arg*
        If *arg* is a ``int`` type, return the arg'th row.
        If *arg* is a ``slice`` of column names, return a new DataFrame with all columns
        sliced to the specified range.
        If *arg* is a ``slice`` of ints, return a new Column or DataFrame with all rows
        sliced to the specified range.
        If *arg* is an ``list`` containing column names, return a new
        DataFrame with the corresponding columns.
        If *arg* is an ``list`` containing row numbers, return a new
        Column/DataFrame with the corresponding rows.
        If *arg* is a ``BooleanColumn``, return a new Column or DataFrame
        with rows which have been  marked True
        """

        if isinstance(arg, int):
            return self.get(arg)
        elif isinstance(arg, str):
            return self.get_column(arg)
        elif isinstance(arg, slice):
            args = []
            for i in [arg.start, arg.stop, arg.step]:
                if isinstance(i, np.integer):
                    args.append(int(i))
                else:
                    args.append(i)
            if all(a is None or isinstance(a, int) for a in args):
                return self.slice(*args)
            elif all(a is None or isinstance(a, str) for a in args):
                if arg.step is not None:
                    raise TypeError(
                        f"column slice can't have step argument {arg.step}")
                return self.slice_columns(arg.start, arg.stop)
            else:
                raise TypeError(
                    f"slice arguments {[type(a) for a in args]} should be ints or strings")
        elif isinstance(arg, (tuple, list)):
            if len(arg) == 0:
                return self
            if all(isinstance(a, int) for a in arg):
                return self.gets(arg)
            if all(isinstance(a, str) for a in arg):
                return self.get_columns(arg)
            if all(isinstance(a, bool) for a in arg):
                return self.filter(arg)
            else:
                raise TypeError("index should be list of int or list of str")
        elif isinstance(arg, AbstractColumn) and is_boolean(arg.dtype):
            return self.filter(arg)
        else:
            raise self._not_supported("__getitem__")

    def gets(self, indices):
        # default implementation; optimized ones in subclasses
        res = self._Empty(self.dtype)
        for i in indices:
            (m, d) = (self.getmask(i), self.getdata(i))
            if m:
                res._append_null()
            else:
                res._append_value(d)
        return res._finalize()

    def slice(self, start, stop, step):
        # default implementation; optimized ones in subclasses
        res = self._Empty(self.dtype)
        for i in list(range(len(self)))[start:stop:step]:
            (m, d) = (self.getmask(i), self.getdata(i))
            if m:
                res._append_null()
            else:
                res._append_value(d)
        return res._finalize()

    def get_column(self, column):
        raise self._not_supported('get_column')

    def get_columns(self, columns):
        raise self._not_supported('get_columns')

    def slice_columns(self, start, stop):
        raise self._not_supported('slice_columns')

    @trace
    @expression
    def head(self, n=5):
        """Return the first `n` rows."""
        return self[:n]

    @trace
    @expression
    def tail(self, n=5):
        """Return the last `n` rows."""
        return self[-n:]

    @trace
    @expression
    def reverse(self):
        return self[::-1]

    # iterators  -------------------------------------------------------------

    def __iter__(self):
        """Return the iterator object itself."""
        for i in range(len(self)):
            yield self.get(i)

    def items(self):
        for i in range(len(self)):
            yield (self.getmask(i), self.getdata(i))

    def _vectorize(self, fun, dtype):
        # note: vectorize preserves mask!
        default = dtype.default_value()
        res = self.session._Empty(dtype, mask=self._mask)
        for m, i in self.items():
            if m:
                res._append_data(default)
            else:
                res._append_data(fun(i))
        return res._finalize()

    # conversions -------------------------------------------------------------

    # @abstractmethod

    def astype(self, dtype):
        """Cast the Column to the given dtype"""
        raise self._not_supported('astype')

    # functools map/filter/reduce ---------------------------------------------

    @ trace
    @ expression
    def map(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None] = None,
        dtype: Optional[DType] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Maps rows according to input correspondence.
        dtype required if result type != item type.
        """
        # to avoid applying the function to missing values, use
        #   na_action == 'ignore'
        if columns is not None:
            raise TypeError(
                f"columns parameter for flat columns not supported")
        dtype = dtype if dtype is not None else self._dtype
        res = self._Empty(dtype)
        if isinstance(arg, defaultdict):
            for masked, i in self.items():
                if not masked:
                    res._append(arg[i])
                elif na_action is None:
                    res._append(arg[None])
                else:
                    res._append_null()
        elif isinstance(arg, dict):
            for masked, i in self.items():
                if not masked:
                    if i in arg:
                        res._append(arg[i])
                    else:
                        res._append_null()
                elif None in arg and na_action is None:
                    res._append(arg[None])
                else:
                    res._append_null()
            return res._finalize()
        else:  # arg must be a function
            for masked, i in self.items():
                if not masked:
                    res._append(arg(i))
                elif na_action is None:
                    res._append(arg(None))
                else:  # na_action == 'ignore'
                    res._append_null()
        return res._finalize()

    @ trace
    @ expression
    def flatmap(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None] = None,
        dtype: Optional[DType] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Maps rows to list of rows according to input correspondance
        dtype required if result type != item type.
        """
        if columns is not None:
            raise TypeError(
                f"columns parameter for flat columns not supported")

        def func(x):
            return arg.get(x, None) if isinstance(arg, dict) else arg(x)

        dtype_ = dtype if dtype is not None else self._dtype
        res = self._Empty(dtype_)
        for masked, i in self.items():
            if not masked:
                res._extend(func(i))
            elif na_action is None:
                res._extend(func(None))
            else:
                res._append_null()
        return res._finalize()

    @ trace
    @ expression
    def filter(
        self, predicate: Union[Callable, Iterable], columns: Optional[List[str]] = None
    ):
        """
        Select rows where predicate is True.
        Different from Pandas. Use keep for Pandas filter.
        """
        if columns is not None:
            raise TypeError(
                f"columns parameter for flat columns not supported")

        if not isinstance(predicate, Iterable) and not callable(predicate):
            raise TypeError(
                "predicate must be a unary boolean predicate or iterable of booleans"
            )
        res = self._Empty(self._dtype)
        if callable(predicate):
            for x in self:
                if predicate(x):
                    res._append(x)
        elif isinstance(predicate, Iterable):
            for x, p in zip(self, predicate):
                if p:
                    res._append(x)
        else:
            pass
        return res._finalize()

    @trace
    @expression
    def reduce(self, fun, initializer=None, finalizer=None):
        """
        Apply binary function cumulatively to the rows[0:],
        so as to reduce the column/dataframe to a single value
        """
        if len(self) == 0:
            if initializer is not None:
                return initializer
            else:
                raise TypeError(
                    "reduce of empty sequence with no initial value")
        start = 0
        if initializer is None:
            value = self[0]
            start = 1
        else:
            value = initializer
        for i in range(start, len(self)):
            value = fun(value, self[i])
        if finalizer is not None:
            return finalizer(value)
        else:
            return value

    # ifthenelse -----------------------------------------------------------------

    def iif(self, then_, else_):
        """Equivalent to ternary expression: if self then then_ else else_"""
        raise self._not_supported('astype')
    # def where(self, condition, other):
    #     """Equivalent to ternary expression: if condition then self else other"""
    #     if not isinstance(condition, Iterable):
    #         raise TypeError("condition must be an iterable of booleans")

    #     res = self._empty(self._dtype)
    #     # check for length?
    #     if isinstance(other, ScalarTypeValues):
    #         for s, m in zip(self, condition):
    #             if m:
    #                 res._append(s)
    #             else:
    #                 res._append(other)
    #         return res
    #     elif isinstance(other, Iterable):
    #         for s, m, o in zip(self, condition, other):
    #             if m:
    #                 res._append(s)
    #             else:
    #                 res._append(o)
    #         return res
    #     else:
    #         raise TypeError(f"where on type {type(other)} is not supported")

    # def mask(self, condition, other):
    #     """Equivalent to ternary expression: if ~condition then self else other."""
    #     return self.where(~condition, other)

    # sorting and top-k -------------------------------------------------------

    @trace
    @expression
    def sort(
        self,
        by: Optional[List[str]] = None,
        ascending=True,
        na_position: Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order"""
        raise self._not_supported('astype')

    @trace
    @expression
    def nlargest(
        self,
        n=5,
        columns: Optional[List[str]] = None,
        keep: Literal["last", "first"] = "first",
    ):
        """Returns a new data of the *n* largest element."""
        # keep="all" not supported
        if columns is not None:
            raise TypeError(
                "computing n-largest on non-structured column can't have 'columns' parameter"
            )
        return self.sort(ascending=False).head(n)

    @trace
    @expression
    def nsmallest(
        self, n=5, columns: Optional[List[str]] = None, keep="first"
    ):
        """Returns a new data of the *n* smallest element. """
        # keep="all" not supported
        if columns is not None:
            raise TypeError(
                "computing n-smallest on non-structured column can't have 'columns' parameter"
            )

        return self.sort(ascending=True).head(n)

    @trace
    @expression
    def nunique(self, dropna=True):
        """Returns the number of unique values of the column"""
        raise self._not_supported('astype')

    # operators ---------------------------------------------------------------

    # # isin --------------------------------------------------------------------
    # @trace
    # @expression
    # def isin(self, values: Union[list, dict, AbstractColumn]):
    #     """Check whether list values are contained in data, or column/dataframe (row/column specific)."""
    #     if isinstance(values, list):
    #         res = self._empty(boolean)
    #         for i in self:
    #             res._append(i in values)
    #         return res
    #     else:
    #         raise ValueError(
    #             f"isin undefined for values of type {type(self).__name__}."
    #         )

    # # data cleaning -----------------------------------------------------------

    # # universal  ---------------------------------------------------------------

    # @trace
    # @expression
    # def min(self, numeric_only=None):
    #     """Return the minimum of the nonnull values of the Column."""
    #     # skipna == True
    #     # default implmentation:
    #     if numeric_only is None or (numeric_only and is_numerical(self.dtype)):
    #         return min(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"min undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def max(self, numeric_only=None):
    #     """Return the maximum of the nonnull values of the column."""
    #     # skipna == True
    #     if numeric_only is None or (numeric_only and is_numerical(self.dtype)):
    #         return max(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"max undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def all(self, boolean_only=None):
    #     """Return whether all nonull elements are True in Column"""
    #     # skipna == True
    #     if boolean_only is None or (boolean_only and is_boolean(self.dtype)):
    #         return all(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"all undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def any(self, skipna=True, boolean_only=None):
    #     """Return whether any nonull element is True in Column"""
    #     # skipna == True
    #     if boolean_only is None or (boolean_only and is_boolean(self.dtype)):
    #         return any(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"all undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def sum(self):
    #     """Return sum of all nonull elements in Column"""
    #     # skipna == True
    #     # only_numerical == True
    #     if is_numerical(self.dtype):
    #         return sum(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"max undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def prod(self):
    #     """Return produce of the values in the data"""
    #     # skipna == True
    #     # only_numerical == True
    #     if is_numerical(self.dtype):
    #         return functools.reduce(operator.mul, self._iter(skipna=True), 1)
    #     else:
    #         raise ValueError(f"prod undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def cummin(self, skipna=True):
    #     """Return cumulative minimum of the data."""
    #     # skipna == True
    #     if is_numerical(self.dtype):
    #         return self._accumulate_column(min, skipna=skipna, initial=None)
    #     else:
    #         raise ValueError(f"cumin undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def cummax(self, skipna=True):
    #     """Return cumulative maximum of the data."""
    #     if is_numerical(self.dtype):
    #         return self._accumulate_column(max, skipna=skipna, initial=None)
    #     else:
    #         raise ValueError(f"cummax undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def cumsum(self, skipna=True):
    #     """Return cumulative sum of the data."""
    #     if is_numerical(self.dtype):
    #         return self._accumulate_column(operator.add, skipna=skipna, initial=None)
    #     else:
    #         raise ValueError(f"cumsum undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def cumprod(self, skipna=True):
    #     """Return cumulative product of the data."""
    #     if is_numerical(self.dtype):
    #         return self._accumulate_column(operator.mul, skipna=skipna, initial=None)
    #     else:
    #         raise ValueError(f"cumprod undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def mean(self):
    #     """Return the mean of the values in the series."""
    #     if is_numerical(self.dtype):
    #         return statistics.mean(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"mean undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def median(self):
    #     """Return the median of the values in the data."""
    #     if is_numerical(self.dtype):
    #         return statistics.median(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"median undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def mode(self):
    #     """Return the mode(s) of the data."""
    #     if is_numerical(self.dtype):
    #         return statistics.mode(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"mode undefined for {type(self).__name__}.")

    # @trace
    # @expression
    # def std(self):
    #     """Return the stddev(s) of the data."""
    #     if is_numerical(self.dtype):
    #         return statistics.stdev(self._iter(skipna=True))
    #     else:
    #         raise ValueError(f"std undefined for {type(self).__name__}.")

    # @trace
    # def _iter(self, skipna):
    #     for i in self:
    #         if not (i is None and skipna):
    #             yield i

    # @trace
    # def _accumulate_column(self, func, *, skipna=True, initial=None):
    #     it = iter(self)
    #     res = self._empty(self.dtype)
    #     total = initial
    #     rest_is_null = False
    #     if initial is None:
    #         try:
    #             total = next(it)
    #         except StopIteration:
    #             raise ValueError(f"cum[min/max] undefined for empty column.")
    #     if total is None:
    #         raise ValueError(f"cum[min/max] undefined for columns with row 0 as null.")
    #     res._append(total)
    #     for element in it:
    #         if rest_is_null:
    #             res._append(None)
    #             continue
    #         if element is None:
    #             if skipna:
    #                 res._append(None)
    #             else:
    #                 res._append(None)
    #                 rest_is_null = True
    #         else:
    #             total = func(total, element)
    #             res._append(total)
    #     return res

    # # describe ----------------------------------------------------------------
    # @trace
    # @expression
    # def describe(
    #     self,
    #     percentiles=None,
    #     include_columns: Union[List[DType], Literal[None]] = None,
    #     exclude_columns: Union[List[DType], Literal[None]] = None,
    # ):
    #     """Generate descriptive statistics."""
    #     from .dataframe import DataFrame

    #     # Not supported: datetime_is_numeric=False,
    #     if include_columns is not None or exclude_columns is not None:
    #         raise TypeError(
    #             f"'include/exclude columns' parameter for '{type(self).__name__}' not supported "
    #         )
    #     if percentiles is None:
    #         percentiles = [25, 50, 75]
    #     percentiles = sorted(set(percentiles))
    #     if len(percentiles) > 0:
    #         if percentiles[0] < 0 or percentiles[-1] > 100:
    #             raise ValueError("percentiles must be betwen 0 and 100")

    #     if is_numerical(self.dtype):
    #         res = DataFrame(
    #             Struct([Field("statistic", string), Field("value", float64)])
    #         )
    #         res._append(("count", self.count()))
    #         res._append(("mean", self.mean()))
    #         res._append(("std", self.std()))
    #         res._append(("min", self.min()))
    #         values = self._percentiles(percentiles)
    #         for p, v in zip(percentiles, values):
    #             res._append((f"{p}%", v))
    #         res._append(("max", self.max()))
    #         return res
    #     else:
    #         raise ValueError(f"median undefined for {type(self).__name__}.")

    # def _percentiles(self, percentiles):
    #     if len(self) == 0 or len(percentiles) == 0:
    #         return []
    #     out = []
    #     s = sorted(self)
    #     for percent in percentiles:
    #         k = (len(self) - 1) * (percent / 100)
    #         f = math.floor(k)
    #         c = math.ceil(k)
    #         if f == c:
    #             out.append(s[int(k)])
    #             continue
    #         d0 = s[int(f)] * (c - k)
    #         d1 = s[int(c)] * (k - f)
    #         out.append(d0 + d1)
    #     return out

    # # Flat column specfic ops ----------------------------------------------------------
    # @trace
    # @expression
    # def is_unique(self):
    #     """Return boolean if data values are unique."""
    #     seen = set()
    #     return not any(i in seen or seen.add(i) for i in self)

    # # only on flat column
    # @trace
    # @expression
    # def is_monotonic_increasing(self):
    #     """Return boolean if values in the object are monotonic increasing"""
    #     return self._compare(operator.lt, initial=True)

    # @trace
    # @expression
    # def is_monotonic_decreasing(self):
    #     """Return boolean if values in the object are monotonic decreasing"""
    #     return self._compare(operator.gt, initial=True)

    # def _compare(self, op, initial):
    #     assert initial in [True, False]
    #     if self._length == 0:
    #         return initial
    #     it = iter(self)
    #     start = next(it)
    #     for step in it:
    #         if op(start, step):
    #             start = step
    #             continue
    #         else:
    #             return False
    #     return True

    # @staticmethod
    # def _flatten(a):
    #     return functools.reduce(operator.iconcat, a, [])

    # # interop ----------------------------------------------------------------

    # @trace
    # def to_pandas(self):
    #     """Convert selef to pandas dataframe"""
    #     # TODO Add type translation
    #     # Skipping analyzing 'pandas': found module but no type hints or library stubs
    #     import pandas as pd  # type: ignore

    #     return pd.Series(self)

    # @trace
    # def to_arrow(self):
    #     """Convert selef to pandas dataframe"""
    #     # TODO Add type translation
    #     import pyarrow as pa  # type: ignore

    #     return pa.array(self)

    # @trace
    # def to_python(self):
    #     """Convert to plain Python container (list of scalars or containers)"""
    #     raise NotImplementedError()


# windows ---------------------------------------------------------------

# def rolling(
#         self, window, min_periods=None, center=False, axis=0, win_type=None
#     ):
#         return Rolling(
#             self,
#             window,
#             min_periods=min_periods,
#             center=center,
#             axis=axis,
#             win_type=win_type,
#         )

    @ expression
    def __eq__(self, other):
        raise self._not_supported('==')

    # windows ---------------------------------------------------------------
    def _valid_mask(len):
        return np.full((len,), False, dtype=np.bool_)
