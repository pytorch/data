#!/usr/bin/env python3
from __future__ import annotations

import abc
import functools
import itertools
import math
import operator
import statistics
import typing as ty
from collections import OrderedDict, defaultdict

import numpy as np
from tabulate import tabulate

import torcharrow.dtypes as dt

from .column_factory import Device
from .expression import expression
from .scope import Scope
from .trace import trace, traceproperty

# ------------------------------------------------------------------------------
# Column Factory with default scope and device


def Column(
    data: ty.Union[ty.Iterable, dt.DType, ty.Literal[None]] = None,
    dtype: ty.Optional[dt.DType] = None,
    scope: ty.Optional[Scope] = None,
    to: Device = "",
):
    """
    Column Factory method; allocates memory on to (or Scope.default.to) device
    """
    scope = scope or Scope.default
    to = to or scope.to
    return scope.Column(data, dtype=dtype, to=to)


# ------------------------------------------------------------------------------
# IColumn


class IColumn(ty.Sized, ty.Iterable, abc.ABC):
    """Interface for Column are n vectors (n>=1) of columns"""

    def __init__(self, scope, to, dtype):

        self._scope = scope
        self._to = to
        self._dtype = dtype

        # id handling, used for tracing...
        self.id = f"c{scope.ct.next()}"

    # getters ---------------------------------------------------------------

    @property
    def scope(self):
        return self._scope

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

    # private builders -------------------------------------------------------

    def _EmptyColumn(self, dtype, mask=None):
        """PRIVATE Column factory; must be follwed by _append... and _finalize"""
        return self.scope._EmptyColumn(dtype, self.to, mask)

    def _FullColumn(self, data, dtype=None, mask=None):
        """PRIVATE Column factory; data must be in the expected representation"""
        return self.scope._FullColumn(data, dtype, self.to, mask)

    def _Column(self, data=None, dtype: ty.Optional[dt.DType] = None):
        """PRIVATE Column factory; must be follwed by _append... and _finalize"""
        return self.scope.Column(data, dtype, to=self.to)

    def _DataFrame(self, data, dtype=None, columns=None):
        """PRIVATE Column factory; data must be in the expected representation"""
        return self.scope.DataFrame(data, dtype, columns, to=self.to)

    # private builders -------------------------------------------------------

    @trace
    @abc.abstractmethod
    def _append_null(self):
        """PRIVATE _append null value with updateing mask"""
        raise self._not_supported("_append_null")

    @trace
    @abc.abstractmethod
    def _append_value(self, value):
        """PRIVATE _append non-null value with updateing mask"""
        raise self._not_supported("_append_value")

    @trace
    @abc.abstractmethod
    def _append_data(self, value):
        """PRIVATE _append non-null value without updating mask"""
        raise self._not_supported("_append_data")

    @trace
    def _append(self, value):
        """PRIVATE _append value"""
        if value is None:
            self._append_null()
        else:
            self._append_value(value)

    def _extend(self, values):
        """PRIVATE _extend values"""
        for value in values:
            self._append(value)

    # public append/copy/astype------------------------------------------------

    @trace
    def append(self, values):
        """Returns column/dataframe with values appended."""
        # TODO use _column_copy, but for now this works...
        res = self._EmptyColumn(self.dtype)
        for (m, d) in self.items():
            if m:
                res._append_null()
            else:
                res._append_value(d)
        for i in values:
            res._append(i)
        return res._finalize()

    @trace
    def concat(self, columns):
        """Returns concatenated columns."""
        # TODO use _column_copy, but for now this works...
        res = self._EmptyColumn(self.dtype)
        for each in [self] + columns:
            for (m, d) in each.items():
                if m:
                    res._append_null()
                else:
                    res._append_value(d)
        return res._finalize()

    @trace
    def copy(self):
        # TODO implement this generically over columns using _FullColumn
        raise self._not_supported("copy")

    def astype(self, dtype):
        """Cast the Column to the given dtype"""
        if dt.is_primitive(self.dtype):
            if dt.is_primitive(dtype):
                fun = dt.cast_as(dtype)
                res = self._emptyColumn(dtype)
                for m, i in self.item():
                    if m:
                        res._append_null()
                    else:
                        res.append_value(fun(i))
                return res._finalize()
            else:
                raise TypeError('f"{astype}({dtype}) is not supported")')
        raise TypeError('f"{astype} for {type(self).__name__} is not supported")')

    # public simple observers -------------------------------------------------

    @trace
    @expression
    def count(self):
        """Return number of non-NA/null observations pgf the column/frame"""
        return len(self) - self.null_count()

    @trace
    @expression
    @abc.abstractmethod
    def null_count(self):
        """Return number of null values"""
        raise self._not_supported("getmask")

    @trace
    @expression
    @abc.abstractmethod
    def __len__(self):
        """Return number of rows including null values"""
        raise self._not_supported("__len__")

    @trace
    @expression
    def length(self):
        """Return number of rows including null values"""
        return len(self)

    # printing ----------------------------------------------------------------

    def __str__(self):
        return f"Column([{', '.join(str(i) for i in self)}], id = {self.id})"

    def __repr__(self):
        rows = [[l if l is not None else "None"] for l in self]
        tab = tabulate(
            rows,
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self._dtype}, length: {len(self)}, null_count: {self.null_count()}"
        return tab + dt.NL + typ

    # private helpers ---------------------------------------------------------

    def _not_supported(self, name):
        raise TypeError(f"{name} for type {type(self).__name__} is not supported")

    scalar_types = (int, float, bool, str)

    # selectors/getters -------------------------------------------------------

    @abc.abstractmethod
    def getmask(self, i):
        """Return mask at index i"""
        raise self._not_supported("getmask")

    @abc.abstractmethod
    def getdata(self, i):
        """Return data at index i"""
        raise self._not_supported("getdata")

    def valid(self, index):
        """Return wether data ast index i is valid, i.e., non-masked"""
        return not self.getmask(index)

    def get(self, index, fill_value=None):
        """Return data[index] or fill_value if data[i] not valid"""
        if self.getmask(index):
            return fill_value
        else:
            return self.getdata(index)

    @trace
    @expression
    def __getitem__(self, arg):
        """
        If *arg* is a

        `n`, a number, return the row with index n
        `[n1,..,nm]` return a new column with the rows[n1],..,rows[nm]
        `[n1:n2:n3]`, return a new column slice with rows[n1:n2:n3]

        `s`, a string, return the column named s
        `[s1,..,sm]` return  dataframe having column[s1],..,column[sm]
        `[s1:s2]` return dataframe having columns[s1:s2]

        `[b1,..,bn]`, where bi are booleans, return all rows that are true
        `Column([b1..bn])` return all rows that are true
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
                    raise TypeError(f"column slice can't have step argument {arg.step}")
                return self.slice_columns(arg.start, arg.stop)
            else:
                raise TypeError(
                    f"slice arguments {[type(a) for a in args]} should all be int or string"
                )
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
        elif isinstance(arg, IColumn) and dt.is_boolean(arg.dtype):
            return self.filter(arg)
        else:
            raise self._not_supported("__getitem__")

    def gets(self, indices):
        """Return a new column with the rows[indices[0]],..,rows[indices[-1]]"""
        res = self._EmptyColumn(self.dtype)
        for i in indices:
            (m, d) = (self.getmask(i), self.getdata(i))
            if m:
                res._append_null()
            else:
                res._append_value(d)
        return res._finalize()

    def slice(self, start, stop, step):
        """Return a new column with the slice rows[start:stop:step]"""
        res = self._EmptyColumn(self.dtype)
        for i in list(range(len(self)))[start:stop:step]:
            (m, d) = (self.getmask(i), self.getdata(i))
            if m:
                res._append_null()
            else:
                res._append_value(d)
        return res._finalize()

    def get_column(self, column):
        """Return the named column"""
        raise self._not_supported("get_column")

    def get_columns(self, columns):
        """Return a new dataframe referencing the columns[s1],..,column[sm]"""
        raise self._not_supported("get_columns")

    def slice_columns(self, start, stop):
        """Return a new dataframe with the slice rows[start:stop]"""
        raise self._not_supported("slice_columns")

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
        """Iterator returning mask,data pairs for all items of a column"""
        for i in range(len(self)):
            yield (self.getmask(i), self.getdata(i))

    def data(self, fill_value=None):
        """Iterator returning non-null or fill_value data of a column"""
        for m, i in self.items():
            if m:
                if fill_value is not None:
                    yield fill_value
            else:
                yield i

    def _vectorize(self, fun, dtype):
        # note: vectorize preserves mask!
        default = dtype.default_value()
        res = self._EmptyColumn(dtype)
        for m, i in self.items():
            if m:
                res._append_null()
            else:
                res._append_value(fun(i))
        return res._finalize()

    # functools map/filter/reduce ---------------------------------------------

    @trace
    @expression
    def map(
        self,
        arg: ty.Union[ty.Dict, ty.Callable],
        na_action: ty.Literal["ignore", None] = None,
        dtype: ty.Optional[dt.DType] = None,
        columns: ty.Optional[ty.List[str]] = None,
    ):
        """
        Maps rows according to input correspondence.
        dtype required if result type != item type.
        """
        # to avoid applying the function to missing values, use
        #   na_action == 'ignore'
        if columns is not None:
            raise TypeError(f"columns parameter for flat columns not supported")
        dtype = dtype if dtype is not None else self._dtype
        res = self._EmptyColumn(dtype)
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

    @trace
    @expression
    def flatmap(
        self,
        arg: ty.Union[ty.Dict, ty.Callable],
        na_action: ty.Literal["ignore", None] = None,
        dtype: ty.Optional[dt.DType] = None,
        columns: ty.Optional[ty.List[str]] = None,
    ):
        """
        Maps rows to list of rows according to input correspondence
        dtype required if result type != item type.
        """

        if columns is not None:
            raise TypeError(f"columns parameter for flat columns not supported")

        def func(x):
            return arg.get(x, None) if isinstance(arg, dict) else arg(x)

        dtype = dtype or self.dtype
        res = self._EmptyColumn(dtype)
        for masked, i in self.items():
            if not masked:
                res._extend(func(i))
            elif na_action is None:
                res._extend(func(None))
            else:
                res._append_null()
        return res._finalize()

    @trace
    @expression
    def filter(
        self,
        predicate: ty.Union[ty.Callable, ty.Iterable],
        columns: ty.Optional[ty.List[str]] = None,
    ):
        """
        Select rows where predicate is True.
        Different from Pandas. Use keep for Pandas filter.
        """
        if columns is not None:
            raise TypeError(f"columns parameter for flat columns not supported")

        if not isinstance(predicate, ty.Iterable) and not callable(predicate):
            raise TypeError(
                "predicate must be a unary boolean predicate or iterable of booleans"
            )
        res = self._EmptyColumn(self._dtype)
        if callable(predicate):
            for x in self:
                if predicate(x):
                    res._append(x)
        elif isinstance(predicate, ty.Iterable):
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
                raise TypeError("reduce of empty sequence with no initial value")
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
        res = self._EmptyColumn(lub)
        for (m, b), t, e in zip(self.items(), then_, else_):
            if m:
                res._append_null()
            elif b:
                res._append(t)
            else:
                res._append(e)
        return res._finalize()

    # sorting and top-k -------------------------------------------------------

    @trace
    @expression
    def sort(
        self,
        by: ty.Optional[ty.List[str]] = None,
        ascending=True,
        na_position: ty.Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order"""
        if by is not None:
            raise TypeError("sorting a non-structured column can't have 'by' parameter")
        res = self._EmptyColumn(self.dtype)
        if na_position == "first":
            res._extend([None] * self.null_count())
        res._extend(sorted((i for i in self if i is not None), reverse=not ascending))
        if na_position == "last":
            res._extend([None] * self.null_count())
        return res._finalize()

    @trace
    @expression
    def nlargest(
        self,
        n=5,
        columns: ty.Optional[ty.List[str]] = None,
        keep: ty.Literal["last", "first"] = "first",
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
    def nsmallest(self, n=5, columns: ty.Optional[ty.List[str]] = None, keep="first"):
        """Returns a new data of the *n* smallest element."""
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
        if not dropna:
            return len(set(self))
        else:
            return len(set(i for i in self if i is not None))

    # operators ---------------------------------------------------------------
    @staticmethod
    def swap(op):
        return lambda a, b: op(b, a)

    @trace
    @expression
    def __add__(self, other):
        """Vectorized a + b."""
        return self._arithmetic_op(other, operator.add)

    @trace
    @expression
    def __radd__(self, other):
        """Vectorized b + a."""
        return self._arithmetic_op(other, IColumn.swap(operator.add))

    @trace
    @expression
    def __sub__(self, other):
        """Vectorized a - b."""
        return self._arithmetic_op(other, operator.sub)

    @trace
    @expression
    def __rsub__(self, other):
        """Vectorized b - a."""
        return self._arithmetic_op(other, IColumn.swap(operator.sub))

    @trace
    @expression
    def __mul__(self, other):
        """Vectorized a * b."""
        return self._arithmetic_op(other, operator.mul)

    @trace
    @expression
    def __rmul__(self, other):
        """Vectorized b * a."""
        return self._arithmetic_op(other, IColumn.swap(operator.mul))

    @trace
    @expression
    def __floordiv__(self, other):
        """Vectorized a // b."""
        return self._arithmetic_op(other, operator.floordiv)

    @trace
    @expression
    def __rfloordiv__(self, other):
        """Vectorized b // a."""
        return self._arithmetic_op(other, IColumn.swap(operator.floordiv))

    @trace
    @expression
    def __truediv__(self, other):
        """Vectorized a / b."""
        return self._arithmetic_op(other, operator.truediv, div="__truediv__")

    @trace
    @expression
    def __rtruediv__(self, other):
        """Vectorized b / a."""
        return self._arithmetic_op(
            other, IColumn.swap(operator.truediv), div="__rtruediv__"
        )

    @trace
    @expression
    def __mod__(self, other):
        """Vectorized a % b."""
        return self._arithmetic_op(other, operator.mod)

    @trace
    @expression
    def __rmod__(self, other):
        """Vectorized b % a."""
        return self._arithmetic_op(other, IColumn.swap(operator.mod))

    @trace
    @expression
    def __pow__(self, other):
        """Vectorized a ** b."""
        return self._arithmetic_op(other, operator.pow)

    @trace
    @expression
    def __rpow__(self, other):
        """Vectorized b ** a."""
        return self._arithmetic_op(other, IColumn.swap(operator.pow))

    @trace
    @expression
    def __eq__(self, other):
        """Vectorized a == b."""
        return self._logic_op(other, operator.eq)

    @trace
    @expression
    def __ne__(self, other):
        """Vectorized a != b."""
        return self._logic_op(other, operator.ne)

    @trace
    @expression
    def __lt__(self, other):
        """Vectorized a < b."""
        return self._logic_op(other, operator.le)

    @trace
    @expression
    def __gt__(self, other):
        """Vectorized a > b."""
        return self._logic_op(other, operator.gt)

    @trace
    @expression
    def __le__(self, other):
        """Vectorized a < b."""
        return self._logic_op(other, operator.le)

    @trace
    @expression
    def __ge__(self, other):
        """Vectorized a < b."""
        return self._logic_op(other, operator.ge)

    @staticmethod
    def _lor(a, b):
        return a or b

    @staticmethod
    def _land(a, b):
        return a and b

    @trace
    @expression
    def __or__(self, other):
        """Vectorized boolean or: a | b."""
        return self._logic_op(other, IColumn._lor)

    @trace
    @expression
    def __ror__(self, other):
        """Vectorized boolean reverse or: b | a."""
        return other._compare(other, IColumn.swap(IColumn._lor))

    @trace
    @expression
    def __and__(self, other):
        """Vectorized boolean and: a & b."""
        return self._logic_op(other, IColumn._land)

    @trace
    @expression
    def __rand__(self, other):
        """Vectorized boolean reverse and: b & a."""
        return self._logic_op(other, IColumn.swap(IColumn._land))

    @trace
    @expression
    def __invert__(self):
        """Vectorized boolean not: ~ a."""
        return self._vectorize(operator.not_, dt.Boolean(self.dtype.nullable))

    @trace
    @expression
    def __neg__(self):
        """Vectorized: - a."""
        return self._vectorize(operator.neg, self.dtype)

    @trace
    @expression
    def __pos__(self):
        """Vectorized: + a."""
        return self._vectorize(operator.pos, self.dtype)

    @staticmethod
    def _isin(values):
        return lambda value: value in values

    @trace
    @expression
    def isin(self, values: ty.Union[list, IColumn, dict]):
        """Check whether values are contained in column."""
        # note mask is True
        res = self._EmptyColumn(dt.boolean, mask=None)
        for m, i in self.items():
            if m:
                res._append_value(False)
            else:
                res._append_value(i in values)
        return res._finalize()

    @trace
    @expression
    def abs(self):
        """Absolute value of each element of the series."""
        return self._vectorize(abs, self.dtype)

    @trace
    @expression
    def ceil(self):
        """Rounds each value upward to the smallest integral"""
        return self._vectorize(math.ceil, self.dtype)

    @trace
    @expression
    def floor(self):
        """Rounds each value downward to the largest integral value"""
        return self._vectorize(math.floor, self.dtype)

    @trace
    @expression
    def round(self, decimals=0):
        """Round each value in a data to the given number of decimals."""
        _round = lambda i: round(i, decimals)
        return self._vectorize(_round, self.dtype)

    def _arithmetic_op(self, other, fun, div=""):
        others = None
        other_dtype = None
        if isinstance(other, IColumn):
            others = other.items()
            other_dtype = other.dtype
        else:
            others = itertools.repeat((False, other))
            other_dtype = dt.infer_dtype_from_value(other)
        if dt.is_boolean_or_numerical(self.dtype) and dt.is_boolean_or_numerical(
            other_dtype
        ):
            if div != "":
                res_dtype = dt.Float64(self.dtype.nullable or other_dtype.nullable)
                res = self._EmptyColumn(res_dtype)
                for (m, i), (n, j) in zip(self.items(), others):
                    # TODO Use error handling to mke this more efficient..
                    if m or n:
                        res._append_null()
                    elif div == "__truediv__" and j == 0:
                        res._append_null()
                    elif div == "__rtruediv__" and i == 0:
                        res._append_null()
                    else:
                        res._append_value(fun(i, j))
                return res._finalize()
            else:
                res_dtype = dt.promote(self.dtype, other_dtype)
                res = self._EmptyColumn(res_dtype)
                for (m, i), (n, j) in zip(self.items(), others):
                    if m or n:
                        res._append_null()
                    else:
                        res._append_value(fun(i, j))
                return res._finalize()
        raise TypeError(f"{type(self).__name__}.{fun.__name__} is not supported")

    def _logic_op(self, other, pred):
        others = None
        other_dtype = None
        if isinstance(other, IColumn):
            others = other.items()
            other_dtype = other.dtype
        else:
            others = itertools.repeat((False, other))
            other_dtype = dt.infer_dtype_from_value(other)
        res = self._EmptyColumn(dt.Boolean(self.dtype.nullable or other_dtype.nullable))
        for (m, i), (n, j) in zip(self.items(), others):
            if m or n:
                res._append_null()
            else:
                res._append_value(pred(i, j))
        return res._finalize()

    # data cleaning -----------------------------------------------------------

    @trace
    @expression
    def fillna(self, fill_value: ty.Union[dt.ScalarTypes, ty.Dict]):
        """Fill NA/NaN values using the specified method."""
        if not isinstance(fill_value, IColumn.scalar_types):
            raise TypeError(f"fillna with {type(fill_value)} is not supported")
        if isinstance(fill_value, IColumn.scalar_types):
            res = self._EmptyColumn(self.dtype.constructor(nullable=False), mask=None)
            for m, i in self.items():
                if not m:
                    res._append_value(i)
                else:
                    res._append_value(fill_value)
            return res._finalize()
        else:
            raise TypeError(f"fillna with {type(fill_value)} is not supported")

    @trace
    @expression
    def dropna(self, how: ty.Literal["any", "all"] = "any"):
        """Return a column/frame with rows removed where a row has any or all
        nulls."""
        if dt.is_primitive(self.dtype):
            res = self._EmptyColumn(self.dtype.constructor(nullable=False), mask=None)
            for m, i in self.items():
                if not m:
                    res._append_value(i)
            return res._finalize()
        else:
            raise TypeError(f"dropna for type {self.dtype} is not supported")

    @trace
    @expression
    def drop_duplicates(
        self,
        subset: ty.Union[str, ty.List[str], ty.Literal[None]] = None,
        keep: ty.Literal["first", "last", False] = "first",
    ):
        """Remove duplicate values from row/frame but keep the first, last, none"""
        # TODO Add functionality for first and last
        assert keep == "first"
        if subset is not None:
            raise TypeError(f"subset parameter for flat columns not supported")
        res = self._EmptyColumn(self._dtype)
        res._extend(list(OrderedDict.fromkeys(self)))
        return res._finalize()

    # # universal  ---------------------------------------------------------------

    def _check(self, pred, name):
        if not pred(self.dtype):
            raise ValueError(f"{name} undefined for {type(self).__name__}.")

    @trace
    @expression
    def min(self, fill_value=None):
        """Return the minimum of the non-null values."""
        return min(self.data(fill_value))

    @trace
    @expression
    def max(self, fill_value=None):
        """Return the maximum of the non-null values."""
        return max(self.data(fill_value))

    @trace
    @expression
    def all(self):
        """Return whether all non-null elements are True"""
        return all(self.data())

    @trace
    @expression
    def any(self, skipna=True):
        """Return whether any non-null element is True in Column"""
        return any(self.data())

    @trace
    @expression
    def sum(self):
        """Return sum of all non-null elements"""
        self._check(dt.is_numerical, "sum")
        return sum(self.data())

    @trace
    @expression
    def prod(self):
        """Return produce of the non-null values in the data"""
        self._check(dt.is_numerical, "prod")
        return functools.reduce(operator.mul, self.data(), 1)

    @trace
    @expression
    def cummin(self):
        """Return cumulative minimum of the data."""
        return self._accumulate(min)

    @trace
    @expression
    def cummax(self):
        """Return cumulative maximum of the data."""
        return self._accumulate(max)

    @trace
    @expression
    def cumsum(self):
        """Return cumulative sum of the data."""
        self._check(dt.is_numerical, "cumsum")
        return self._accumulate(operator.add)

    @trace
    @expression
    def cumprod(self):
        """Return cumulative product of the data."""
        self._check(dt.is_numerical, "cumprod")
        return self._accumulate(operator.mul)

    @trace
    @expression
    def mean(self):
        self._check(dt.is_numerical, "mean")
        """Return the mean of the non-null values in the series."""
        m = statistics.mean((float(i) for i in list(self.data())))
        return m

    @trace
    @expression
    def median(self):
        """Return the median of the values in the data."""
        self._check(dt.is_numerical, "median")
        return statistics.median((float(i) for i in list(self.data())))

    @trace
    @expression
    def mode(self):
        """Return the mode(s) of the data."""
        self._check(dt.is_numerical, "mode")
        return statistics.mode(self.data())

    @trace
    @expression
    def std(self):
        """Return the stddev(s) of the data."""
        self._check(dt.is_numerical, "std")
        return statistics.stdev((float(i) for i in list(self.data())))

    def _accumulate(self, func):
        total = None
        res = self._EmptyColumn(self.dtype)
        for m, i in self.items():
            if m:
                res._append_null()
            elif total is None:
                res._append_value(i)
                total = i
            else:
                total = func(total, i)
                res._append_value(total)
        m = res._finalize()
        return m

    @trace
    @expression
    def percentiles(self, q, interpolation="midpoint"):
        """Compute the q-th percentile of non-null data."""
        if interpolation != "midpoint":
            raise TypeError(
                f"percentiles for '{type(self).__name__}' with parameter other than 'midpoint' not supported "
            )
        if len(self) == 0 or len(q) == 0:
            return []
        out = []
        s = sorted(self)
        for percent in q:
            k = (len(self) - 1) * (percent / 100)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                out.append(s[int(k)])
                continue
            d0 = s[int(f)] * (c - k)
            d1 = s[int(c)] * (k - f)
            out.append(d0 + d1)
        return out

    # describe ----------------------------------------------------------------
    @trace
    @expression
    def describe(
        self,
        percentiles_=None,
        include_columns: ty.Union[ty.List[dt.DType], ty.Literal[None]] = None,
        exclude_columns: ty.Union[ty.List[dt.DType], ty.Literal[None]] = None,
    ):
        """Generate descriptive statistics."""
        import torcharrow.idataframe

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

        if dt.is_numerical(self.dtype):
            res = self._EmptyColumn(
                dt.Struct(
                    [dt.Field("statistic", dt.string), dt.Field("value", dt.float64)]
                )
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
            raise ValueError(f"describe undefined for {type(self).__name__}.")

    # unique and montonic -----------------------------------------------------
    @trace
    @expression
    def is_unique(self):
        """Return boolean if data values are unique."""
        seen = set()
        return not any(i in seen or seen.add(i) for i in self)

    # only on flat column
    @trace
    @expression
    def is_monotonic_increasing(self):
        """Return boolean if values in the object are monotonic increasing"""
        return self._compare(operator.lt, initial=True)

    @trace
    @expression
    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonic decreasing"""
        return self._compare(operator.gt, initial=True)

    def _compare(self, op, initial):
        assert initial in [True, False]
        if len(self) == 0:
            return initial
        it = iter(self)
        start = next(it)
        for step in it:
            if step is None:
                continue
            if op(start, step):
                start = step
                continue
            else:
                return False
        return True

    # interop ----------------------------------------------------------------

    @trace
    def to_pandas(self):
        """Convert self to pandas dataframe"""
        import pandas as pd  # type: ignore

        # default implementation, normally this should be zero copy...
        return pd.Series(self)

    @trace
    def to_arrow(self):
        """Convert self to pandas dataframe"""
        import pyarrow as pa  # type: ignore

        # default implementation, normally this should be zero copy...
        return pa.array(self)

    # batching/unbatching -----------------------------------------------------
    # NOTE experimental
    def batch(self, n):
        assert n > 0
        i = 0
        while i < len(self):
            h = i
            i = i + n
            yield self[h:i]

    @staticmethod
    def unbatch(iter: ty.Iterable[IColumn]):
        res = []
        for i in iter:
            res.append(i)
        if len(res) == 0:
            raise ValueError("can't determine column type")
        return res[0].concat(res[1:])
