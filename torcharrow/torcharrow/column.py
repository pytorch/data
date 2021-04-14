#!/usr/bin/env python3
from __future__ import annotations

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

# ------------------------------------------------------------------------------
# Column - the factory method for specialized columns.


@trace
@expression
def Column(
    data: Union[Iterable, DType, Literal[None]] = None, dtype: Optional[DType] = None
):
    """
    Column factory method; returned column type has elements of homogenous dtype.
    """

    # handling cyclic references...
    from .list_column import ListColumn
    from .map_column import MapColumn
    from .numerical_column import NumericalColumn
    from .string_column import StringColumn

    if data is None and dtype is None:
        raise TypeError("Column requires data and/or dtype parameter")

    if data is not None and isinstance(data, DType):
        if dtype is not None and isinstance(dtype, DType):
            raise TypeError("Column can only have one dtype parameter")
        dtype = data
        data = None

    # dtype given, optional data
    if dtype is not None:
        if isinstance(dtype, DType):
            col = _column_constructor(dtype)
            if data is not None:
                for i in data:
                    col.append(i)
            return col
        else:
            raise TypeError(
                f"dtype parameter of DType expected (got {type(dtype).__name__})"
            )

    # data given, optional column
    if data is not None:
        if isinstance(data, Sequence):
            data = iter(data)
        if isinstance(data, Iterable):
            prefix = []
            for i, v in enumerate(data):
                prefix.append(v)
                if i > 5:
                    break
            dtype = infer_dtype_from_prefix(prefix)
            if dtype is None:
                raise TypeError("Column cannot infer type from data")
            if is_tuple(dtype):
                raise TypeError(
                    "Column cannot be used to created structs, use Dataframe constructor instead"
                )
            col = _column_constructor(dtype)
            # add prefix and ...
            col.extend(prefix)
            # ... continue enumerate the data
            for _, v in enumerate(data):
                col.append(v)
            return col
        else:
            raise TypeError(
                f"data parameter of Sequence type expected (got {type(dtype).__name__})"
            )
    else:
        raise AssertionError("unexpected case")


def arange(
    start: int,
    stop: int,
    step: int = 1,
    dtype: Optional[DType] = None,
) -> AbstractColumn:
    return Column(list(range(start, stop, step)), dtype)


# - underlying factory ---------------------------------------------------------
_factory: List[Tuple[Callable, Callable]] = []

# called once by each concrete Column to add this to the factory...


def _set_column_constructor(test, constructor):
    _factory.append((test, constructor))


def _column_constructor(dtype, kwargs=None):
    for test, constructor in _factory:
        if test(dtype):
            return constructor(dtype, kwargs)
    raise KeyError(f"no matching test found for {dtype}")


# ------------------------------------------------------------------------------
# AbstractColumn


class AbstractColumn(ABC, Sized, Iterable):
    """AbstractColumn are one dimenensionalcolumns or two dimensional dataframes"""

    _ct = 0  # global counter for all columns/dataframes ever created...

    def __init__(self, dtype: Optional[DType]):
        # id handling, used for tracing...
        self.id = f"c{AbstractColumn._ct}"
        AbstractColumn._ct += 1
        # normal constructor code
        self._dtype = dtype
        self._offset = 0
        self._length = 0
        self._null_count = 0
        self._validity = ar.array("b")

    @classmethod
    def reset(cls):
        cls._ct = 0

    # simple meta data getters -----------------------------------------------

    @property
    @traceproperty
    def dtype(self):
        """dtype of the colum/frame"""
        return self._dtype

    @property
    @traceproperty
    def isnullable(self):
        """A boolean indicating whether column/frame can have nulls"""
        return self.dtype.nullable

    @abstractproperty
    def is_appendable(self):
        """Can this column/frame be extended without side effecting """
        pass

    @trace
    @expression
    def count(self):
        """Return number of non-NA/null observations pgf the column/frame"""
        return self._length - self._null_count

    @trace
    @expression
    def null_count(self):
        """Number of null values"""
        return self._null_count

    @trace
    @expression
    def __len__(self):
        """Return number of rows including null values"""
        return self._length

    @property
    @traceproperty
    def ndim(self):
        """Column ndim is always 1, Frame ndim is always 2"""
        return 1

    @property
    @traceproperty
    def size(self):
        """Number of rows * number of columns."""
        return len(self)

    @abstractmethod
    def _raw_lengths(self) -> List[int]:
        "Lengths of underlying buffers"
        pass

    def _valid(self, i):
        return self._validity[self._offset + i]

    # builders and iterators---------------------------------------------------
    @trace
    def append(self, value):
        """Append value to the end of the column/frame"""
        if not self.is_appendable:
            raise AttributeError("column is not appendable")
        else:
            return self._append(value)

    @abstractmethod
    def _append(self, value):
        pass

    def extend(self, iterable: Iterable):
        """Append items from iterable to the end of the column/frame"""
        if not self.is_appendable:
            raise AttributeError("column is not appendable")
        for i in iterable:
            self._append(i)

    def concat(self, others: List["AbstractColumn"]):
        """Concatenate columns/frames."""
        # instead of pandas.concat
        for o in others:
            self.extend(o)

    @trace
    def copy(self, deep=True):
        """Make a copy of this object’s description and if deep also its data."""
        return self._copy(deep, self._offset, self._length)

    @abstractmethod
    def _copy(self, deep, offset, length):
        pass

    # selectors -----------------------------------------------------------
    @abstractmethod
    def get(self, arg, fill_value):
        """Get ith row from column/frame"""
        pass

    @abstractmethod
    def __iter__(self):
        """Return the iterator object itself."""
        pass

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
        # print('slice', arg, str(type(arg)))
        if isinstance(arg, int):
            if arg < 0:
                arg = arg + len(self)
            return self._get_row(arg)
        elif isinstance(arg, str):
            return self._get_column(arg)
        elif isinstance(arg, slice):
            args = [arg.start, arg.stop, arg.step]
            if all(a is None or isinstance(a, str) for a in args):
                return self._slice_columns(arg)
            elif all(a is None or isinstance(a, int) for a in args):
                return self._slice_rows(arg)
            else:
                TypeError("slice arguments should be ints or strings")
        if isinstance(arg, (tuple, list)):
            if len(arg) == 0:
                return self._empty()
            elif isinstance(arg[0], str) and all(isinstance(a, str) for a in arg):
                return self._pick_columns(arg)
            elif isinstance(arg[0], int) and all(isinstance(a, int) for a in arg):
                return self._pick_rows(arg)
            else:
                raise TypeError("index should be list of int or list of str")
            return self._get_columns_by_label(arg, downcast=True)

        elif isinstance(arg, AbstractColumn) and is_boolean(arg.dtype):
            return self.filter(arg)
        else:
            raise TypeError(f"__getitem__ on type {type(arg)} is not supported")

    @trace
    @expression
    def slice(self, start=None, stop=None, step=1):
        return self._slice_rows(slice(start, stop, step))

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
        res = _column_constructor(self.dtype)
        for i in range(len(self)):
            res._append(self[(len(self) - 1) - i])
        return res

    def _pick_columns(self, arg):
        raise AttributeError(f"{type(self)} has no columns to select from")

    def _get_row(self, arg, default=None):
        return self.get(arg, default)

    def _pick_rows(self, arg):
        res = _column_constructor(self.dtype)
        for i in arg:
            res._append(self[i])
        return res

    def _slice_rows(self, arg):
        # TODO This is Python Slice! (Pandas last is inclusive, Python it is exclusive)

        start, stop, step = arg.indices(len(self))

        if start <= stop and step == 1:
            # create a view
            res = self.copy(deep=False)
            res._offset = res._offset + start
            res._length = stop - start
            # update null_count
            res._nullcount = sum(self._valid(i) for i in range(len(self)))
            return res
        else:
            # usual slice
            res = _column_constructor(self.dtype)
            for i in range(start, stop, step):
                res._append(self.get(i, None))
            return res

    def _get_column(self, arg, default=None):
        raise AttributeError(f"{type(self)} has no column to get")

    def _slice_columns(self, slice):
        raise AttributeError(f"{type(self)} has no column to slice")

    # conversions -------------------------------------------------------------
    # TODO
    def to_frame(self, name: Optional[str]):
        """Convert data into a DataFrame"""
        raise NotImplementedError()

    #     if name is not None:
    #         col = name
    #     elif self.name is None:
    #         col = 'f0'
    #     else:
    #         col = self.name
    #     return DataFrame({col: self})

    # def to_array(self, fillna=None):
    #     """Get a dense numpy array for the data."""

    def astype(self, dtype):
        """Cast the Column to the given dtype"""
        raise NotImplementedError()
        # if dtype.isconvertable_from(self.dtype):
        #     # TODO
        #     # create new stuff
        #     pass
        # else:
        #     raise TypeError(f"'cannot cast from 'self.dtype' to '{dtype}'")

    # functools map/filter/reduce ---------------------------------------------
    @trace
    @expression
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
        if columns is not None:
            raise TypeError(f"columns parameter for flat columns not supported")

        def func(x):
            return arg.get(x, None) if isinstance(arg, dict) else arg(x)

        dtype = dtype if dtype is not None else self._dtype

        res = _column_constructor(dtype)
        for i in range(self._length):
            if self._valid(i) or na_action == "ignore":
                res._append(func(self[i]))
            else:
                res._append(None)
        return res

    @trace
    @expression
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
            raise TypeError(f"columns parameter for flat columns not supported")

        def func(x):
            return arg.get(x, None) if isinstance(arg, dict) else arg(x)

        dtype1 = dtype if dtype is not None else self._dtype
        res = _column_constructor(dtype1)
        for i in range(self._length):
            if self._valid(i) or na_action == "ignore":
                res.extend(func(self[i]))
            else:
                res._append(None)
        return res

    @trace
    @expression
    def filter(
        self, predicate: Union[Callable, Iterable], columns: Optional[List[str]] = None
    ):
        """
        Select rows where predicate is True.
        Different from Pandas. Use keep for Pandas filter.
        """
        if columns is not None:
            raise TypeError(f"columns parameter for flat columns not supported")

        if not isinstance(predicate, Iterable) and not callable(predicate):
            raise TypeError(
                "predicate must be a unary boolean predicate or iterable of booleans"
            )
        res = _column_constructor(self._dtype)
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
        return res

    @trace
    @expression
    def reduce(self, fun, initializer=None):
        """
        Apply binary function cumulatively to the rows[0:],
        so as to reduce the column/dataframe to a single value
        """
        if self._length == 0:
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
        for i in range(start, self._length):
            value = fun(value, self[i])
        return value

    # ifthenelse -----------------------------------------------------------------

    # def where(self, condition, other):
    #     """Equivalent to ternary expression: if condition then self else other"""
    #     if not isinstance(condition, Iterable):
    #         raise TypeError("condition must be an iterable of booleans")

    #     res = _column_constructor(self._dtype)
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

    # sorting -----------------------------------------------------------------

    @trace
    @expression
    def sort_values(
        self,
        by: Union[str, List[str], Literal[None]] = None,
        ascending=True,
        na_position: Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order"""
        # key:Callable, optional missing
        if by is not None:
            raise TypeError("sorting a non-structured column can't have 'by' parameter")
        res = _column_constructor(self.dtype)
        if na_position == "first":
            res.extend([None] * self._null_count)
        res.extend(sorted((i for i in self if i is not None), reverse=not ascending))
        if na_position == "last":
            res.extend([None] * self._null_count)
        return res

    @trace
    @expression
    def nlargest(
        self,
        n=5,
        columns: Union[str, List[str], Literal[None]] = None,
        keep: Literal["last", "first"] = "first",
    ):
        """Returns a new data of the *n* largest element."""
        # keep="all" not supported
        # pass columns for comparison , nlargest(by=['country', ..]..)
        # first/last matters only for dataframes with columns
        # Todo add keep arg
        if columns is not None:
            raise TypeError(
                "computing n-largest on non-structured column can't have 'columns' parameter"
            )
        return self.sort_values(ascending=False).head(n)

    @trace
    @expression
    def nsmallest(
        self, n=5, columns: Union[str, List[str], Literal[None]] = None, keep="first"
    ):
        """Returns a new data of the *n* smallest element. """
        # keep="all" not supported
        # first/last matters only for dataframes with columns
        # Todo add keep arg
        if columns is not None:
            raise TypeError(
                "computing n-smallest on non-structured column can't have 'columns' parameter"
            )

        return self.sort_values(ascending=True).head(n)

    @trace
    @expression
    def nunique(self, dropna=True):
        """Returns the number of unique values of the column"""
        if not dropna:
            return len(set(self))
        else:
            return len(set(i for i in self if i is not None))

    # operators ---------------------------------------------------------------

    # arithmetic

    @expression
    def add(self, other, fill_value=None):
        return self._binary_operator("add", other, fill_value=fill_value)

    @expression
    def radd(self, other, fill_value=None):
        return self._binary_operator("add", other, fill_value=fill_value, reflect=True)

    @expression
    def __add__(self, other):
        return self._binary_operator("add", other)

    @expression
    def __radd__(self, other):
        return self._binary_operator("add", other, reflect=True)

    @expression
    def sub(self, other, fill_value=None):
        return self._binary_operator("sub", other, fill_value=fill_value)

    @expression
    def rsub(self, other, fill_value=None):
        return self._binary_operator("sub", other, fill_value=fill_value, reflect=True)

    @expression
    def __sub__(self, other):
        return self._binary_operator("sub", other)

    @expression
    def __rsub__(self, other):
        return self._binary_operator("sub", other, reflect=True)

    @expression
    def mul(self, other, fill_value=None):
        return self._binary_operator("mul", other, fill_value=fill_value)

    @expression
    def rmul(self, other, fill_value=None):
        return self._binary_operator("mul", other, fill_value=fill_value, reflect=True)

    @expression
    def __mul__(self, other):
        return self._binary_operator("mul", other)

    @expression
    def __rmul__(self, other):
        return self._binary_operator("mul", other, reflect=True)

    @expression
    def floordiv(self, other, fill_value=None):
        return self._binary_operator("floordiv", other, fill_value=fill_value)

    @expression
    def rfloordiv(self, other, fill_value=None):
        return self._binary_operator(
            "floordiv", other, fill_value=fill_value, reflect=True
        )

    @expression
    def __floordiv__(self, other):
        return self._binary_operator("floordiv", other)

    @expression
    def __rfloordiv__(self, other):
        return self._binary_operator("floordiv", other, reflect=True)

    @expression
    def truediv(self, other, fill_value=None):
        return self._binary_operator(
            "truediv",
            other,
            fill_value=fill_value,
        )

    @expression
    def rtruediv(self, other, fill_value=None):
        return self._binary_operator(
            "truediv", other, fill_value=fill_value, reflect=True
        )

    @expression
    def __truediv__(self, other):
        return self._binary_operator("truediv", other)

    @expression
    def __rtruediv__(self, other):
        return self._binary_operator("truediv", other, reflect=True)

    @expression
    def mod(self, other, fill_value=None):
        return self._binary_operator("mod", other, fill_value=fill_value)

    @expression
    def rmod(self, other, fill_value=None):
        return self._binary_operator("mod", other, fill_value=fill_value, reflect=True)

    @expression
    def __mod__(self, other):
        return self._binary_operator("mod", other)

    @expression
    def __rmod__(self, other):
        return self._binary_operator("mod", other, reflect=True)

    @expression
    def pow(self, other, fill_value=None):
        return self._binary_operator(
            "pow",
            other,
            fill_value=fill_value,
        )

    @expression
    def rpow(self, other, fill_value=None):
        return self._binary_operator("pow", other, fill_value=fill_value, reflect=True)

    @expression
    def __pow__(self, other):
        return self._binary_operator("pow", other)

    @expression
    def __rpow__(self, other):
        return self._binary_operator("pow", other, reflect=True)

    # comparison

    @expression
    def eq(self, other, fill_value=None):
        return self._binary_operator("eq", other, fill_value=fill_value)

    @expression
    def __eq__(self, other):
        return self._binary_operator("eq", other)

    @expression
    def ne(self, other, fill_value=None):
        return self._binary_operator("ne", other, fill_value=fill_value)

    @expression
    def __ne__(self, other):
        return self._binary_operator("ne", other)

    @expression
    def lt(self, other, fill_value=None):
        return self._binary_operator("lt", other, fill_value=fill_value)

    @expression
    def __lt__(self, other):
        return self._binary_operator("lt", other)

    @expression
    def gt(self, other, fill_value=None):
        return self._binary_operator("gt", other, fill_value=fill_value)

    @expression
    def __gt__(self, other):
        return self._binary_operator("gt", other)

    @expression
    def le(self, other, fill_value=None):
        return self._binary_operator("le", other, fill_value=fill_value)

    @expression
    def __le__(self, other):
        return self._binary_operator("le", other)

    @expression
    def ge(self, other, fill_value=None):
        return self._binary_operator("ge", other, fill_value=fill_value)

    @expression
    def __ge__(self, other):
        return self._binary_operator("ge", other)

    # bitwise or (|), used for logical or
    @expression
    def __or__(self, other):
        return self._binary_operator("or", other)

    @expression
    def __ror__(self, other):
        return self._binary_operator("or", other, reflect=True)

    # bitwise and (&), used for logical and
    @expression
    def __and__(self, other):
        return self._binary_operator("and", other)

    @expression
    def __rand__(self, other):
        return self._binary_operator("and", other, reflect=True)

    # bitwise complement (~), used for logical not

    @expression
    def __invert__(self):
        return self._unary_operator(operator.__not__, self.dtype)

    # unary arithmetic
    @expression
    def __neg__(self):
        return self._unary_operator(operator.neg, self.dtype)

    @expression
    def __pos__(self):
        return self._unary_operator(operator.pos, self.dtype)

    # unary math related

    @expression
    def abs(self):
        """Absolute value of each element of the series."""
        return self._unary_operator(abs, self.dtype)

    @expression
    def ceil(self):
        """Rounds each value upward to the smallest integral"""
        return self._unary_operator(math.ceil, Int64(self.dtype.nullable))

    @expression
    def floor(self):
        """Rounds each value downward to the largest integral value"""
        return self._unary_operator(math.floor, Int64(self.dtype.nullable))

    @expression
    def round(self, decimals: Optional[int] = None):
        """Round each value in a data to the given number of decimals."""
        if decimals is None:
            return self._unary_operator(round, Int64(self.dtype.nullable))
        else:

            def round_(x):
                return round(x, decimals)

            return self._unary_operator(round_, Float64(self.dtype.nullable))

    @expression
    def hash_values(self):
        """Compute the hash of values in this column."""
        return self._unary_operator(hash, Int64(self.dtype.nullable))

    @trace
    def _broadcast(self, operator, const, fill_value, dtype, reflect):
        assert is_primitive(self._dtype) and is_primitive(dtype)
        res = _column_constructor(dtype)
        # TODO: introduce fast paths...
        for i in range(self._length):
            if self._valid(i):
                if reflect:
                    res._append(operator(const, self.get(i, None)))
                else:
                    res._append(operator(self.get(i, None), const))
            elif fill_value is not None:
                if reflect:
                    res._append(operator(const, fill_value))
                else:
                    res._append(operator(fill_value, const))
            else:
                res._append(None)
        return res

    @trace
    def _pointwise(self, operator, other, fill_value, dtype, reflect):
        assert is_primitive(self._dtype) and is_primitive(dtype)
        # print('Column._pointwise', self, operator, other, fill_value, dtype, reflect)
        res = _column_constructor(dtype)
        #  # TODO: introduce fast paths...
        for i in range(self._length):
            if self._valid(i) and other._valid(i):
                if reflect:
                    res._append(operator(other.get(i, None), self.get(i, None)))
                else:
                    res._append(operator(self.get(i, None), other.get(i, None)))
            elif fill_value is not None:
                l = self.get(i, None) if self._valid(i) else fill_value
                r = other.get(i, None) if other._valid(i) else fill_value
                if reflect:
                    res._append(operator(r, l))
                else:
                    res._append(operator(l, r))
            else:
                res._append(None)
        return res

    @trace
    def _unary_operator(self, operator, dtype):
        res = _column_constructor(dtype)
        for i in range(self._length):
            if self._valid(i):
                res._append(operator(self[i]))
            else:
                res._append(None)
        return res

    @trace
    def _binary_operator(self, operator, other, fill_value=None, reflect=False):
        if isinstance(other, (int, float, list, set, type(None))):
            return self._broadcast(
                derive_operator(operator),
                other,
                fill_value,
                derive_dtype(self._dtype, operator),
                reflect,
            )
        elif is_struct(other.dtype):
            raise TypeError(
                f"cannot apply '{operator}' on {type(self).__name__} and {type(other).__name__}"
            )
        else:
            return self._pointwise(
                derive_operator(operator),
                other,
                fill_value,
                derive_dtype(self._dtype, operator),
                reflect,
            )

    # isin --------------------------------------------------------------------
    @trace
    @expression
    def isin(self, values: Union[list, dict, AbstractColumn]):
        """Check whether list values are contained in data, or column/dataframe (row/column specific)."""
        if isinstance(values, list):
            res = _column_constructor(boolean)
            for i in self:
                res._append(i in values)
            return res
        else:
            raise ValueError(
                f"isin undefined for values of type {type(self).__name__}."
            )

    # data cleaning -----------------------------------------------------------

    @trace
    @expression
    def fillna(
        self, fill_value: Union[ScalarTypes, Dict, AbstractColumn, Literal[None]]
    ):
        """Fill NA/NaN values with scalar (like 0) or column/dataframe (row/column specific)"""
        if fill_value is None:
            return self
        assert self._dtype is not None
        res = _column_constructor(self._dtype.constructor(nullable=False))
        if isinstance(fill_value, ScalarTypeValues):
            for value in self:
                if value is not None:
                    res._append(value)
                else:
                    res._append(fill_value)
            return res
        elif isinstance(
            fill_value, AbstractColumn
        ):  # TODO flat Column   --> needs a test
            for value, fill in zip(self, fill_value):
                if value is not None:
                    res._append(value)
                else:
                    res._append(fill)
            return res
        else:  # Dict and Dataframe only self is dataframe
            raise TypeError(f"fillna with {type(fill_value)} is not supported")

    @trace
    @expression
    def dropna(self, how: Literal["any", "all"] = "any"):
        """Return a column/frame with rows removed where a row has any or all nulls."""
        # TODO only flat columns supported...
        # notet hat any and all play nor role for flat columns,,,
        assert self._dtype is not None
        res = _column_constructor(self._dtype.constructor(nullable=False))
        for i in range(self._offset, self._offset + self._length):
            if self._validity[i]:
                res._append(self[i])
        return res

    @trace
    @expression
    def drop_duplicates(
        self,
        subset: Union[str, List[str], Literal[None]] = None,
        keep: Literal["first", "last", False] = "first",
    ):
        """ Remove duplicate values from row/frame but keep the first, last, none"""
        # Todo Add functionality
        assert keep == "first"
        if subset is not None:
            raise TypeError(f"subset parameter for flat columns not supported")
        res = _column_constructor(self._dtype)
        res.extend(list(OrderedDict.fromkeys(self)))
        return res

    # universal  ---------------------------------------------------------------

    @trace
    @expression
    def min(self, numeric_only=None):
        """Return the minimum of the nonnull values of the Column."""
        # skipna == True
        # default implmentation:
        if numeric_only is None or (numeric_only and is_numerical(self.dtype)):
            return min(self._iter(skipna=True))
        else:
            raise ValueError(f"min undefined for {type(self).__name__}.")

    @trace
    @expression
    def max(self, numeric_only=None):
        """Return the maximum of the nonnull values of the column."""
        # skipna == True
        if numeric_only is None or (numeric_only and is_numerical(self.dtype)):
            return max(self._iter(skipna=True))
        else:
            raise ValueError(f"max undefined for {type(self).__name__}.")

    @trace
    @expression
    def all(self, boolean_only=None):
        """Return whether all nonull elements are True in Column"""
        # skipna == True
        if boolean_only is None or (boolean_only and is_boolean(self.dtype)):
            return all(self._iter(skipna=True))
        else:
            raise ValueError(f"all undefined for {type(self).__name__}.")

    @trace
    @expression
    def any(self, skipna=True, boolean_only=None):
        """Return whether any nonull element is True in Column"""
        # skipna == True
        if boolean_only is None or (boolean_only and is_boolean(self.dtype)):
            return any(self._iter(skipna=True))
        else:
            raise ValueError(f"all undefined for {type(self).__name__}.")

    @trace
    @expression
    def sum(self):
        """Return sum of all nonull elements in Column"""
        # skipna == True
        # only_numerical == True
        if is_numerical(self.dtype):
            return sum(self._iter(skipna=True))
        else:
            raise ValueError(f"max undefined for {type(self).__name__}.")

    @trace
    @expression
    def prod(self):
        """Return produce of the values in the data"""
        # skipna == True
        # only_numerical == True
        if is_numerical(self.dtype):
            return functools.reduce(operator.mul, self._iter(skipna=True), 1)
        else:
            raise ValueError(f"prod undefined for {type(self).__name__}.")

    @trace
    @expression
    def cummin(self, skipna=True):
        """Return cumulative minimum of the data."""
        # skipna == True
        if is_numerical(self.dtype):
            return self._accumulate_column(min, skipna=skipna, initial=None)
        else:
            raise ValueError(f"cumin undefined for {type(self).__name__}.")

    @trace
    @expression
    def cummax(self, skipna=True):
        """Return cumulative maximum of the data."""
        if is_numerical(self.dtype):
            return self._accumulate_column(max, skipna=skipna, initial=None)
        else:
            raise ValueError(f"cummax undefined for {type(self).__name__}.")

    @trace
    @expression
    def cumsum(self, skipna=True):
        """Return cumulative sum of the data."""
        if is_numerical(self.dtype):
            return self._accumulate_column(operator.add, skipna=skipna, initial=None)
        else:
            raise ValueError(f"cumsum undefined for {type(self).__name__}.")

    @trace
    @expression
    def cumprod(self, skipna=True):
        """Return cumulative product of the data."""
        if is_numerical(self.dtype):
            return self._accumulate_column(operator.mul, skipna=skipna, initial=None)
        else:
            raise ValueError(f"cumprod undefined for {type(self).__name__}.")

    @trace
    @expression
    def mean(self):
        """Return the mean of the values in the series."""
        if is_numerical(self.dtype):
            return statistics.mean(self._iter(skipna=True))
        else:
            raise ValueError(f"mean undefined for {type(self).__name__}.")

    @trace
    @expression
    def median(self):
        """Return the median of the values in the data."""
        if is_numerical(self.dtype):
            return statistics.median(self._iter(skipna=True))
        else:
            raise ValueError(f"median undefined for {type(self).__name__}.")

    @trace
    @expression
    def mode(self):
        """Return the mode(s) of the data."""
        if is_numerical(self.dtype):
            return statistics.mode(self._iter(skipna=True))
        else:
            raise ValueError(f"mode undefined for {type(self).__name__}.")

    @trace
    @expression
    def std(self):
        """Return the stddev(s) of the data."""
        if is_numerical(self.dtype):
            return statistics.stdev(self._iter(skipna=True))
        else:
            raise ValueError(f"std undefined for {type(self).__name__}.")

    @trace
    def _iter(self, skipna):
        for i in self:
            if not (i is None and skipna):
                yield i

    @trace
    def _accumulate_column(self, func, *, skipna=True, initial=None):
        it = iter(self)
        res = _column_constructor(self.dtype)
        total = initial
        rest_is_null = False
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                raise ValueError(f"cum[min/max] undefined for empty column.")
        if total is None:
            raise ValueError(f"cum[min/max] undefined for columns with row 0 as null.")
        res._append(total)
        for element in it:
            if rest_is_null:
                res._append(None)
                continue
            if element is None:
                if skipna:
                    res._append(None)
                else:
                    res._append(None)
                    rest_is_null = True
            else:
                total = func(total, element)
                res._append(total)
        return res

    # describe ----------------------------------------------------------------
    @trace
    @expression
    def describe(
        self,
        percentiles=None,
        include_columns: Union[List[DType], Literal[None]] = None,
        exclude_columns: Union[List[DType], Literal[None]] = None,
    ):
        """Generate descriptive statistics."""
        from .dataframe import DataFrame

        # Not supported: datetime_is_numeric=False,
        if include_columns is not None or exclude_columns is not None:
            raise TypeError(
                f"'include/exclude columns' parameter for '{type(self).__name__}' not supported "
            )
        if percentiles is None:
            percentiles = [25, 50, 75]
        percentiles = sorted(set(percentiles))
        if len(percentiles) > 0:
            if percentiles[0] < 0 or percentiles[-1] > 100:
                raise ValueError("percentiles must be betwen 0 and 100")

        if is_numerical(self.dtype):
            res = DataFrame(
                Struct([Field("statistic", string), Field("value", float64)])
            )
            res._append(("count", self.count()))
            res._append(("mean", self.mean()))
            res._append(("std", self.std()))
            res._append(("min", self.min()))
            values = self._percentiles(percentiles)
            for p, v in zip(percentiles, values):
                res._append((f"{p}%", v))
            res._append(("max", self.max()))
            return res
        else:
            raise ValueError(f"median undefined for {type(self).__name__}.")

    def _percentiles(self, percentiles):
        if len(self) == 0 or len(percentiles) == 0:
            return []
        out = []
        s = sorted(self)
        for percent in percentiles:
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

    # Flat column specfic ops ----------------------------------------------------------
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
        if self._length == 0:
            return initial
        it = iter(self)
        start = next(it)
        for step in it:
            if op(start, step):
                start = step
                continue
            else:
                return False
        return True

    @staticmethod
    def _flatten(a):
        return functools.reduce(operator.iconcat, a, [])

    # interop ----------------------------------------------------------------

    @trace
    def to_pandas(self):
        """Convert selef to pandas dataframe"""
        # TODO Add type translation
        # Skipping analyzing 'pandas': found module but no type hints or library stubs
        import pandas as pd  # type: ignore

        return pd.Series(self)

    @trace
    def to_arrow(self):
        """Convert selef to pandas dataframe"""
        # TODO Add type translation
        import pyarrow as pa  # type: ignore

        return pa.array(self)

    @trace
    def to_python(self):
        """Convert to plain Python container (list of scalars or containers)"""
        raise NotImplementedError()


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
