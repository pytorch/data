#!/usr/bin/env python3
from __future__ import annotations
from abc import abstractmethod

import array as ar
import functools
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from tabulate import tabulate

import torcharrow.dtypes as dt

from torcharrow.column_factory import ColumnFactory, Device
from torcharrow.icolumn import IColumn
from torcharrow.idataframe import IDataFrame
from torcharrow.expression import eval_expression, expression
from torcharrow.scope import Scope

from torcharrow.trace import trace, traceproperty

# assumes that these have been imported already:
# from .inumerical_column import INumericalColumn
# from .istring_column import IStringColumn
# from .imap_column import IMapColumn
# from .ilist_column import IListColumn

# ------------------------------------------------------------------------------
# DataFrame Factory with default scope and device


# -----------------------------------------------------------------------------
# DataFrames aka (StructColumns, can be nested as StructColumns:-)

DataOrDTypeOrNone = Union[Mapping, Sequence, dt.DType, Literal[None]]


class DataFrameStd(IDataFrame):
    """Dataframe, ordered dict of typed columns of the same length"""

    def __init__(self, scope, to, dtype, field_data, mask):
        assert dt.is_struct(dtype)
        super().__init__(scope, to, dtype)

        self._field_data = field_data
        self._mask = mask

    # Any _full requires no further type changes..
    @staticmethod
    def _full(scope, to, data, dtype=None, mask=None):
        cols = data.values()
        assert all(isinstance(c, IColumn) for c in data.values())
        ct = 0
        if len(data) > 0:
            ct = len(list(cols)[0])
            if not all(len(c) == ct for c in cols):
                ValueError(f"length of all columns must be the same (e.g {ct})")
        inferred_dtype = dt.Struct([(n, c.dtype) for n, c in data.items()])
        if dtype is None:
            dtype = inferred_dtype
        else:
            # TODO this must be weakened (to deal with nulls, etc)...
            if dtype != inferred_dtype:
                pass
                # raise TypeError(f'type of data {inferred_dtype} and given type {dtype} must be the same')
        if mask is None:
            mask = DataFrameStd._valid_mask(ct)
        elif len(data) != len(mask):
            raise ValueError(f"data length {len(data)} must be mask length {len(mask)}")
        return DataFrameStd(scope, to, dtype, data, mask)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(scope, to, dtype, mask=None):
        field_data = {f.name: scope._EmptyColumn(f.dtype, to) for f in dtype.fields}
        _mask = mask if mask is not None else ar.array("b")
        return DataFrameStd(scope, to, dtype, field_data, _mask)

    def _append_null(self):
        self._mask.append(True)
        for c in self._field_data.values():
            c._append_null()

    def _append_value(self, value):
        self._mask.append(False)
        for c, v in zip(self._field_data.values(), value):
            c._append(v)

    def _append_data(self, value):
        for c, v in zip(self._field_data.values(), value):
            c._append_data(v)

    def _finalize(self):
        ln = 0
        for c in self._field_data.values():
            _ = c._finalize()
            ln = len(c)
        if isinstance(self._mask, (bool, np.bool8)):
            self._mask = np.full((ln,), self._mask, dtype=np.bool8)
        elif isinstance(self._mask, ar.array):
            self._mask = np.array(self._mask, dtype=np.bool8, copy=False)
        else:
            assert isinstance(self._mask, np.ndarray)
        return self

    def _fromdata(self, field_data, mask=False):
        dtype = dt.Struct([dt.Field(n, c.dtype) for n, c in field_data.items()])
        _mask = mask
        if isinstance(mask, (bool, np.bool8)):
            if len(field_data) == 0:
                _mask = np.full((0,), bool(mask), dtype=np.bool8)
            else:
                n, c = next(iter(field_data.items()))
                _mask = np.full((len(c),), bool(mask), dtype=np.bool8)
        return DataFrameStd(self.scope, self.to, dtype, field_data, _mask)

    def __len__(self):
        if len(self._field_data) == 0:
            return 0
        else:
            n = self.dtype.fields[0].name
            return len(self._field_data[n])

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        return self._mask[i]

    def getdata(self, i):
        return tuple([self._field_data[n].get(i) for n in self.columns])

    @staticmethod
    def _valid_mask(ct):
        return np.full((ct,), False, dtype=np.bool8)

    def append(self, values):
        """Returns column/dataframe with values appended."""
        tmp = self.scope.DataFrame(values, dtype=self.dtype, to=self.to)
        field_data = {n: c.append(tmp[n]) for n, c in self._field_data.items()}
        mask = np.append(self._mask, tmp._mask)
        return self._fromdata(field_data, mask)

    def _check_columns(self, columns):
        for n in columns:
            if n not in self._field_data:
                raise TypeError(f"column {n} not among existing dataframe columns")

    # implementing abstract methods ----------------------------------------------

    @property  # type: ignore
    @traceproperty
    def columns(self):
        """The column labels of the DataFrame."""
        return [f.name for f in self.dtype.fields]

    @trace
    def __setitem__(self, name: str, value: Any) -> None:
        d = None
        if isinstance(value, IColumn):
            d = value
        elif isinstance(value, Iterable):
            d = self.scope.Column(value)
        else:
            raise TypeError("data must be a column or list")

        assert d is not None
        if all(len(c) == 0 for c in self._field_data.values()):
            self._mask = np.full((len(d),), False, dtype=np.bool8)

        elif len(d) != len(self):
            raise TypeError("all columns/lists must have equal length")

        if name in self._field_data.keys():
            raise AttributeError(f"cannot override existing column {name}")
        elif (
            self._dtype is not None
            and isinstance(self._dtype, dt.Struct)
            and len(self.dtype.fields) < len(self._field_data)
        ):
            raise AttributeError("cannot append column to view")
        else:
            assert self._dtype is not None and isinstance(self._dtype, dt.Struct)
            # side effect on field_data
            self._field_data[name] = d
            # no side effect on dtype
            fields = list(self._dtype.fields)
            assert d._dtype is not None
            fields.append(dt.Field(name, d._dtype))
            self._dtype = dt.Struct(fields)

    # printing ----------------------------------------------------------------

    def __str__(self):
        def quote(n):
            return f"'{n}'"

        return f"self._fromdata({dt.OPEN}{', '.join(f'{quote(n)}:{str(c)}' for n,c in self._field_data.items())}, id = {self.id}{dt.CLOSE})"

    def __repr__(self):
        data = []
        for i in self:
            if i is None:
                data.append(["None"] * len(self.columns))
            else:
                assert len(i) == len(self.columns)
                data.append(list(i))
        tab = tabulate(
            data, headers=["index"] + self.columns, tablefmt="simple", showindex=True
        )
        typ = f"dtype: {self._dtype}, count: {self.length()}, null_count: {self.null_count()}"
        return tab + dt.NL + typ

    # selectors -----------------------------------------------------------

    def _column_index(self, arg):
        columns = self.columns
        try:
            return columns.index(arg)
        except ValueError:
            try:
                i = int(arg)
                if 0 <= i and i <= len(columns):
                    return i
                raise TypeError(
                    f"index {i} must be within 0 and less than {len(columns)}"
                )
            except ValueError:
                raise TypeError(f"{arg} must be a column name or column index")

    def gets(self, indices):
        return self._fromdata(
            {n: c[indices] for n, c in self._field_data.items()}, self._mask[indices]
        )

    def slice(self, start, stop, step):
        return self._fromdata(
            {n: c[start:stop:step] for n, c in self._field_data.items()},
            self._mask[start:stop:step],
        )

    def get_column(self, column):
        i = self._column_index(column)
        return self._field_data[self.columns[i]]

    def get_columns(self, columns):
        # TODO: decide on nulls, here we assume all defined (mask = False) for new parent...
        res = {}
        for n in columns:
            m = self.columns[self._column_index(n)]
            res[m] = self._field_data[m]
        return self._fromdata(res, False)

    def slice_columns(self, start, stop):
        # TODO: decide on nulls, here we assume all defined (mask = False) for new parent...
        _start = 0 if start is None else self._column_index(start)
        _stop = len(self.columns) if stop is None else self._column_index(stop)
        res = {}
        for i in range(_start, _stop):
            m = self.columns[i]
            res[m] = self._field_data[m]
        return self._fromdata(res, False)

    # functools map/filter/reduce ---------------------------------------------

    @trace
    @expression
    def map(
        self,
        arg: Union[Dict, Callable],
        /,
        na_action: Literal["ignore", None] = None,
        dtype: Optional[dt.DType] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Maps rows according to input correspondence.
        dtype required if result type != item type.
        """
        if columns is None:
            return super().map(arg, na_action, dtype)
        self._check_columns(columns)

        if len(columns) == 1:
            return self._field_data[columns[0]].map(arg, na_action, dtype)
        else:

            def func(*x):
                return arg.get(tuple(*x), None) if isinstance(arg, dict) else arg(*x)

            dtype = dtype if dtype is not None else self._dtype

            cols = [self._field_data[n] for n in columns]
            res = self._EmptyColumn(dtype)
            for i in range(len(self)):
                if self.valid(i):
                    res._append(func(*[col[i] for col in cols]))
                elif na_action is None:
                    res._append(func(None))
                else:
                    res._append(None)
            return res._finalize()

    @trace
    @expression
    def flatmap(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None] = None,
        dtype: Optional[dt.DType] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Maps rows to list of rows according to input correspondence
        dtype required if result type != item type.
        """
        if columns is None:
            return super().flatmap(arg, na_action, dtype)
        self._check_columns(columns)

        if len(columns) == 1:
            return self._field_data[columns[0]].flatmap(arg, na_action, dtype,)
        else:

            def func(x):
                return arg.get(x, None) if isinstance(arg, dict) else arg(x)

            dtype_ = dtype if dtype is not None else self._dtype
            cols = [self._field_data[n] for n in columns]
            res = self._EmptyColumn(dtype_)
            for i in range(len(self)):
                if self.valid(i):
                    res._extend(func(*[col[i] for col in cols]))
                elif na_action is None:
                    res._extend(func(None))
                else:
                    res._append([])
            return res._finalize()

    @trace
    @expression
    def filter(
        self, predicate: Union[Callable, Iterable], columns: Optional[List[str]] = None
    ):
        """
        Select rows where predicate is True.
        Different from Pandas. Use keep for Pandas filter.
        """
        if columns is None:
            return super().filter(predicate)

        self._check_columns(columns)

        if not isinstance(predicate, Iterable) and not callable(predicate):
            raise TypeError(
                "predicate must be a unary boolean predicate or iterable of booleans"
            )

        res = self._EmptyColumn(self._dtype)
        cols = [self._field_data[n] for n in columns]
        if callable(predicate):
            for i in range(len(self)):
                if predicate(*[col[i] for col in cols]):
                    res._append(self[i])
        elif isinstance(predicate, Iterable):
            for x, p in zip(self, predicate):
                if p:
                    res._append(x)
        else:
            pass
        return res._finalize()

    # sorting ----------------------------------------------------------------

    @trace
    @expression
    def sort(
        self,
        by: Optional[List[str]] = None,
        ascending=True,
        na_position: Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order"""
        # Not allowing None in comparison might be too harsh...
        # Move all rows with None that in sort index to back...
        func = None
        if isinstance(by, list):
            xs = []
            for i in by:
                _ = self._field_data[i]  # throws key error
                xs.append(self.columns.index(i))
            reorder = xs + [j for j in range(len(self._field_data)) if j not in xs]

            def func(tup):
                return tuple(tup[i] for i in reorder)

        res = self._EmptyColumn(self.dtype)
        if na_position == "first":
            res._extend([None] * self.null_count())
        res._extend(
            sorted((i for i in self if i is not None), reverse=not ascending, key=func)
        )
        if na_position == "last":
            res._extend([None] * self.null_count())
        return res._finalize()

    @trace
    @expression
    def nlargest(
        self,
        n=5,
        columns: Optional[List[str]] = None,
        keep: Literal["last", "first"] = "first",
    ):
        """Returns a new dataframe of the *n* largest elements."""
        # Todo add keep arg
        return self.sort(by=columns, ascending=False).head(n)

    @trace
    @expression
    def nsmallest(
        self,
        n=5,
        columns: Optional[List[str]] = None,
        keep: Literal["last", "first"] = "first",
    ):
        """Returns a new dataframe of the *n* smallest elements."""
        return self.sort(by=columns, ascending=True).head(n)

    # operators --------------------------------------------------------------

    @expression
    def __add__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c + other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c + other for (n, c) in self._field_data.items()})

    @expression
    def __radd__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] + c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: other + c for (n, c) in self._field_data.items()})

    @expression
    def __sub__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c - other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c - other for (n, c) in self._field_data.items()})

    @expression
    def __rsub__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] - c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: other - c for (n, c) in self._field_data.items()})

    @expression
    def __mul__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c * other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c * other for (n, c) in self._field_data.items()})

    @expression
    def __rmul__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] * c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: other * c for (n, c) in self._field_data.items()})

    @expression
    def __floordiv__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c // other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata(
                {n: c // other for (n, c) in self._field_data.items()}
            )

    @expression
    def __rfloordiv__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] // c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata(
                {n: other // c for (n, c) in self._field_data.items()}
            )

    @expression
    def __truediv__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c / other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c / other for (n, c) in self._field_data.items()})

    @expression
    def __rtruediv__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] / c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: other / c for (n, c) in self._field_data.items()})

    @expression
    def __mod__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c % other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c % other for (n, c) in self._field_data.items()})

    @expression
    def __rmod__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] % c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: other % c for (n, c) in self._field_data.items()})

    @expression
    def __pow__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c ** other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata(
                {n: c ** other for (n, c) in self._field_data.items()}
            )

    @expression
    def __rpow__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] ** c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata(
                {n: other ** c for (n, c) in self._field_data.items()}
            )

    @expression
    def __eq__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c == other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata(
                {n: c == other for (n, c) in self._field_data.items()}
            )

    @expression
    def __ne__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c == other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata(
                {n: c == other for (n, c) in self._field_data.items()}
            )

    @expression
    def __lt__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c < other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c < other for (n, c) in self._field_data.items()})

    @expression
    def __gt__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c > other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c > other for (n, c) in self._field_data.items()})

    def __le__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c <= other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata(
                {n: c <= other for (n, c) in self._field_data.items()}
            )

    def __ge__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c >= other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata(
                {n: c >= other for (n, c) in self._field_data.items()}
            )

    def __or__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c | other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c | other for (n, c) in self._field_data.items()})

    def __ror__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] | c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: other | c for (n, c) in self._field_data.items()})

    def __and__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: c & other[n] for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: c & other for (n, c) in self._field_data.items()})

    def __rand__(self, other):
        if isinstance(other, DataFrameStd):
            return self._fromdata(
                {n: other[n] & c for (n, c) in self._field_data.items()}
            )
        else:
            return self._fromdata({n: other & c for (n, c) in self._field_data.items()})

    def __invert__(self):
        return self._fromdata({n: ~c for (n, c) in self._field_data.items()})

    def __neg__(self):
        return self._fromdata({n: -c for (n, c) in self._field_data.items()})

    def __pos__(self):
        return self._fromdata({n: +c for (n, c) in self._field_data.items()})

    # isin ---------------------------------------------------------------

    @trace
    @expression
    def isin(self, values: Union[list, dict, IColumn]):
        """Check whether values are contained in data."""
        if isinstance(values, list):
            return self._fromdata(
                {n: c.isin(values) for n, c in self._field_data.items()}
            )
        if isinstance(values, dict):
            self._check_columns(values.keys())
            return self._fromdata(
                {n: c.isin(values[n]) for n, c in self._field_data.items()}
            )
        if isinstance(values, IDataFrame):
            self._check_columns(values.columns)
            return self._fromdata(
                {n: c.isin(values=list(values[n])) for n, c in self._field_data.items()}
            )
        else:
            raise ValueError(
                f"isin undefined for values of type {type(self).__name__}."
            )

    # data cleaning -----------------------------------------------------------

    @trace
    @expression
    def fillna(self, fill_value: Union[dt.ScalarTypes, Dict, Literal[None]]):
        if fill_value is None:
            return self
        if isinstance(fill_value, IColumn.scalar_types):
            return self._fromdata(
                {n: c.fillna(fill_value) for n, c in self._field_data.items()}
            )
        elif isinstance(fill_value, dict):
            return self._fromdata(
                {
                    n: c.fillna(fill_value[n]) if n in fill_value else c
                    for n, c in self._field_data.items()
                }
            )
        else:
            raise TypeError(f"fillna with {type(fill_value)} is not supported")

    @trace
    @expression
    def dropna(self, how: Literal["any", "all"] = "any"):
        """Return a dataframe with rows removed where the row has any or all nulls."""
        # TODO only flat columns supported...
        assert self._dtype is not None
        res = self._EmptyColumn(self._dtype.constructor(nullable=False))
        if how == "any":
            for i in self:
                if not self._has_any_null(i):
                    res._append(i)
        elif how == "all":
            for i in self:
                if not self._has_all_null(i):
                    res._append(i)
        return res._finalize()

    @trace
    @expression
    def drop_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: Literal["first", "last", False] = "first",
    ):
        """Remove duplicate values from data but keep the first, last, none (keep=False)"""
        columns = subset if subset is not None else self.columns
        self._check_columns(columns)

        # TODO fix slow implementation by vectorization,
        # i.e do unique per column and delete when all agree
        # shortcut once no match is found.

        res = self._EmptyColumn(self.dtype)
        indices = [self.columns.index(s) for s in columns]
        seen = set()
        for tup in self:
            row = tuple(tup[i] for i in indices)
            if row in seen:
                continue
            else:
                seen.add(row)
                res._append(tup)
        return res._finalize()

    # @staticmethod
    def _has_any_null(self, tup) -> bool:
        for t in tup:
            if t is None:
                return True
            if isinstance(t, tuple) and self._has_any_null(t):
                return True
        return False

    # @staticmethod
    def _has_all_null(self, tup) -> bool:
        for t in tup:
            if t is not None:
                return False
            if isinstance(t, tuple) and not self._has_all_null(t):
                return False
        return True

    # universal ---------------------------------------------------------

    # TODO Decide on tracing level: If we trace  'min' om a
    # - highlevel then we can use lambdas inside min
    # - lowelevel, i.e call 'summarize', then lambdas have to become
    #   - global functions if they have no state
    #   - dataclasses with an apply function if they have state

    @staticmethod
    def _cmin(c):
        return c.min

    # with static function

    @trace
    @expression
    def min(self):
        """Return the minimum of the non-null values of the Column."""
        return self._summarize(DataFrameStd._cmin)

    # with dataclass function
    # @expression
    # def min(self, numeric_only=None):
    #     """Return the minimum of the non-null values of the Column."""
    #     return self._summarize(_Min(), {"numeric_only": numeric_only})

    # with lambda
    # @expression
    # def min(self, numeric_only=None):
    #     """Return the minimum of the non-null values of the Column."""
    #     return self._summarize(lambda c: c.min, {"numeric_only": numeric_only})

    @trace
    @expression
    def max(self):
        """Return the maximum of the non-null values of the column."""
        # skipna == True
        return self._summarize(lambda c: c.max)

    @trace
    @expression
    def all(self):
        """Return whether all non-null elements are True in Column"""
        return self._summarize(lambda c: c.all)

    @trace
    @expression
    def any(self):
        """Return whether any non-null element is True in Column"""
        return self._summarize(lambda c: c.any)

    @trace
    @expression
    def sum(self):
        """Return sum of all non-null elements in Column"""
        return self._summarize(lambda c: c.sum)

    @trace
    @expression
    def prod(self):
        """Return produce of the values in the data"""
        return self._summarize(lambda c: c.prod)

    @trace
    @expression
    def cummin(self):
        """Return cumulative minimum of the data."""
        return self._lift(lambda c: c.cummin)

    @trace
    @expression
    def cummax(self):
        """Return cumulative maximum of the data."""
        return self._lift(lambda c: c.cummax)

    @trace
    @expression
    def cumsum(self):
        """Return cumulative sum of the data."""
        return self._lift(lambda c: c.cumsum)

    @trace
    @expression
    def cumprod(self):
        """Return cumulative product of the data."""
        return self._lift(lambda c: c.cumprod)

    @trace
    @expression
    def mean(self):
        """Return the mean of the values in the series."""
        return self._summarize(lambda c: c.mean)

    @trace
    @expression
    def median(self):
        """Return the median of the values in the data."""
        return self._summarize(lambda c: c.median)

    @trace
    @expression
    def mode(self):
        """Return the mode(s) of the data."""
        return self._summarize(lambda c: c.mode)

    @trace
    @expression
    def std(self):
        """Return the stddev(s) of the data."""
        return self._summarize(lambda c: c.std)

    @trace
    @expression
    def nunique(self, dropna=True):
        """Returns the number of unique values per column"""
        res = self._EmptyColumn(
            dt.Struct([dt.Field("column", dt.string), dt.Field("nunique", dt.int64)])
        )
        for n, c in self._field_data.items():
            res._append((n, c.nunique(dropna)))
        return res._finalize()

    def _summarize(self, func):
        res = {}
        for n, c in self._field_data.items():
            res[n] = self.scope.Column([func(c)()])
        return self._fromdata(res, False)

    @trace
    def _lift(self, func):
        res = {}
        for n, c in self._field_data.items():
            res[n] = func(c)()
        return self._fromdata(res, False)

    # describe ----------------------------------------------------------------

    @trace
    @expression
    def describe(
        self, percentiles=None, include_columns=None, exclude_columns=None,
    ):
        """Generate descriptive statistics."""
        # Not supported: datetime_is_numeric=False,
        includes = []
        if include_columns is None:
            includes = [
                n for n, c in self._field_data.items() if dt.is_numerical(c.dtype)
            ]
        elif isinstance(include_columns, list):
            includes = [
                n for n, c in self._field_data.items() if c.dtype in include_columns
            ]
        else:
            raise TypeError(
                f"describe with include_columns of type {type(include_columns).__name__} is not supported"
            )

        excludes = []
        if exclude_columns is None:
            excludes = []
        elif isinstance(exclude_columns, list):
            excludes = [
                n for n, c in self._field_data.items() if c.dtype in exclude_columns
            ]
        else:
            raise TypeError(
                f"describe with exclude_columns of type {type(exclude_columns).__name__} is not supported"
            )
        selected = [i for i in includes if i not in excludes]

        if percentiles is None:
            percentiles = [25, 50, 75]
        percentiles = sorted(set(percentiles))
        if len(percentiles) > 0:
            if percentiles[0] < 0 or percentiles[-1] > 100:
                raise ValueError("percentiles must be betwen 0 and 100")

        res = {}
        res["metric"] = self.scope.Column(
            ["count", "mean", "std", "min"] + [f"{p}%" for p in percentiles] + ["max"]
        )
        for s in selected:
            c = self._field_data[s]
            res[s] = self.scope.Column(
                [c.count(), c.mean(), c.std(), c.min()]
                + c.percentiles(percentiles, "midpoint")
                + [c.max()]
            )
        return self._fromdata(res)

    # Dataframe specific ops --------------------------------------------------    #

    @trace
    @expression
    def drop(self, columns: List[str]):
        """
        Returns DataFrame without the removed columns.
        """
        self._check_columns(columns)
        return self._fromdata(
            {n: c for n, c in self._field_data.items() if n not in columns}
        )

    @trace
    @expression
    def keep(self, columns: List[str]):
        """
        Returns DataFrame with the kept columns only.
        """
        self._check_columns(columns)
        return self._fromdata(
            {n: c for n, c in self._field_data.items() if n in columns}
        )

    @trace
    @expression
    def rename(self, column_mapper: Dict[str, str]):
        self._check_columns(column_mapper.keys())
        return self._fromdata(
            {
                column_mapper[n] if n in column_mapper else n: c
                for n, c in self._field_data.items()
            }
        )

    @trace
    @expression
    def reorder(self, columns: List[str]):
        """
        Returns DataFrame with the columns in the prescribed order.
        """
        self._check_columns(columns)
        return self._fromdata({n: self._field_data[n] for n in columns})

    # interop ----------------------------------------------------------------

    def to_pandas(self):
        """Convert self to pandas dataframe"""
        # TODO Add type translation.
        # Skipping analyzing 'pandas': found module but no type hints or library stubs
        import pandas as pd  # type: ignore

        map = {}
        for n, c in self._field_data.items():
            map[n] = c.to_pandas()
        return pd.DataFrame(map)

    def to_arrow(self):
        """Convert self to arrow table"""
        # TODO Add type translation
        import pyarrow as pa  # type: ignore

        map = {}
        for n, c in self._field_data.items():
            map[n] = c.to_arrow()
        return pa.table(map)

    # fluent with symbolic expressions ----------------------------------------

    # TODO decide on whether we nat to have arbitrarily nested wheres...
    @trace
    @expression
    def where(self, *conditions):
        """
        Analogous to SQL's where (NOT Pandas where)

        Filter a dataframe to only include
        rows satisfying a given set of conditions.
        """

        if len(conditions) == 0:
            return self

        values = []
        for i, condition in enumerate(conditions):
            value = eval_expression(condition, {"me": self})
            values.append(value)

        reduced_values = functools.reduce(lambda x, y: x & y, values)
        return self[reduced_values]

    @trace
    @expression
    def select(self, *args, **kwargs):
        """
        Analogous to SQL's ``SELECT`.

        Transform a dataframe by selecting old columns and new (computed)
        columns.
        """

        input_columns = set(self.columns)

        has_star = False
        include_columns = []
        exclude_columns = []
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError("args must be column names")
            if arg == "*":
                if has_star:
                    raise ValueError("select received repeated stars")
                has_star = True
            elif arg in input_columns:
                if arg in include_columns:
                    raise ValueError(
                        f"select received a repeated column-include ({arg})"
                    )
                include_columns.append(arg)
            elif arg[0] == "-" and arg[1:] in input_columns:
                if arg in exclude_columns:
                    raise ValueError(
                        f"select received a repeated column-exclude ({arg[1:]})"
                    )
                exclude_columns.append(arg[1:])
            else:
                raise ValueError(f"argument ({arg}) does not denote an existing column")
        if exclude_columns and not has_star:
            raise ValueError("select received column-exclude without a star")
        if has_star and include_columns:
            raise ValueError("select received both a star and column-includes")
        if set(include_columns) & set(exclude_columns):
            raise ValueError(
                "select received overlapping column-includes and " + "column-excludes"
            )

        include_columns_inc_star = self.columns if has_star else include_columns

        output_columns = [
            col for col in include_columns_inc_star if col not in exclude_columns
        ]

        res = {}
        for n, c in self._field_data.items():
            if n in output_columns:
                res[n] = c
        for n, c in kwargs.items():
            res[n] = eval_expression(c, {"me": self})
        return self._fromdata(res)

    @trace
    @expression
    def pipe(self, func, *args, **kwargs):
        """
        Apply func(self, *args, **kwargs).
        """
        return func(self, *args, **kwargs)

    @trace
    @expression
    def groupby(
        self, by: List[str], sort=False, dropna=True,
    ):
        # TODO implement
        assert not sort
        assert dropna
        self._check_columns(by)

        key_columns = by
        key_fields = []
        item_fields = []
        for k in key_columns:
            key_fields.append(dt.Field(k, self.dtype.get(k)))
        for f in self.dtype.fields:
            if f.name not in key_columns:
                item_fields.append(f)

        groups: Dict[Tuple, ar.array] = {}
        for i in range(len(self)):
            if self.valid(i):
                key = tuple(self._field_data[f.name][i] for f in key_fields)
                if key not in groups:
                    groups[key] = ar.array("I")
                df = groups[key]
                df.append(i)
            else:
                pass
        return GroupedDataFrame(key_fields, item_fields, groups, self)


@dataclass
class GroupedDataFrame:
    _key_fields: List[dt.Field]
    _item_fields: List[dt.Field]
    _groups: Mapping[Tuple, Sequence]
    _parent: DataFrameStd

    @property
    def _scope(self):
        return self._parent._scope

    @property  # type: ignore
    @traceproperty
    def size(self):
        """
        Return the size of each group (including nulls).
        """
        res = self._parent._EmptyColumn(
            dt.Struct(self._key_fields + [dt.Field("size", dt.int64)])
        )
        for k, c in self._groups.items():
            row = k + (len(c),)
            res._append(row)
        return res._finalize()

    def __iter__(self):
        """
        Yield pairs of grouped tuple and the grouped dataframe
        """
        for g, xs in self._groups.items():
            df = self._parent._EmptyColumn(dt.Struct(self._item_fields))
            for x in xs:
                df._append(
                    tuple(
                        self._parent._field_data[f.name][x] for f in self._item_fields
                    )
                )
            yield g, df._finalize()

    @trace
    def _lift(self, op: str) -> IColumn:
        if len(self._key_fields) > 0:
            # it is a dataframe operation:
            return self._combine(op)
        elif len(self._item_fields) == 1:
            return self._apply1(self._item_fields[0], op)
        raise AssertionError("unexpected case")

    def _combine(self, op: str):
        agg_fields = [dt.Field(f"{f.name}.{op}", f.dtype) for f in self._item_fields]
        res = {}
        for f, c in zip(self._key_fields, self._unzip_group_keys()):
            res[f.name] = c
        for f, c in zip(agg_fields, self._apply(op)):
            res[f.name] = c
        return self._parent._fromdata(res)

    def _apply(self, op: str) -> List[IColumn]:
        cols = []
        for f in self._item_fields:
            cols.append(self._apply1(f, op))
        return cols

    def _apply1(self, f: dt.Field, op: str) -> IColumn:
        src_t = f.dtype
        dest_f, dest_t = dt.get_agg_op(op, src_t)
        res = self._parent._EmptyColumn(dest_t)
        src_c = self._parent._field_data[f.name]
        for g, xs in self._groups.items():
            dest_data = [src_c[x] for x in xs]
            dest_c = dest_f(self._parent.scope.Column(dest_data, dtype=dest_t))
            res._append(dest_c)
        return res._finalize()

    def _unzip_group_keys(self) -> List[IColumn]:
        cols = []
        for f in self._key_fields:
            cols.append(self._parent._EmptyColumn(f.dtype))
        for tup in self._groups.keys():
            for i, t in enumerate(tup):
                cols[i]._append(t)
        return [col._finalize() for col in cols]

    def __getitem__(self, arg):
        """
        Return the named grouped column
        """
        # TODO extend that this works inside struct frames as well,
        # e.g. grouped['a']['b'] where grouped returns a struct column having 'b' as field
        if isinstance(arg, str):
            for f in self._item_fields:
                if f.name == arg:
                    return GroupedDataFrame([], [f], self._groups, self._parent)
            for i, f in enumerate(self._key_fields):
                if f.name == arg:
                    res = self._parent._EmptyColumn(f.dtype)
                    for tup in self._groups.keys():
                        res._append(tup[i])
                    return res._finalize()
            raise ValueError(f"no column named ({arg}) in grouped dataframe")
        raise TypeError(f"unexpected type for arg ({type(arg).__name})")

    def min(self, numeric_only=None):
        """Return the minimum of the non-null values of the Column."""
        assert numeric_only == None
        return self._lift("min")

    def max(self, numeric_only=None):
        """Return the minimum of the non-null values of the Column."""
        assert numeric_only == None
        return self._lift("min")

    def all(self, boolean_only=None):
        """Return whether all non-null elements are True in Column"""
        # skipna == True
        return self._lift("all")

    def any(self, skipna=True, boolean_only=None):
        """Return whether any non-null element is True in Column"""
        # skipna == True
        return self._lift("any")

    def sum(self):
        """Return sum of all non-null elements in Column"""
        # skipna == True
        # only_numerical == True
        # skipna == True
        return self._lift("sum")

    def prod(self):
        """Return produce of the values in the data"""
        # skipna == True
        # only_numerical == True
        return self._lift("prod")

    def mean(self):
        """Return the mean of the values in the series."""
        return self._lift("mean")

    def median(self):
        """Return the median of the values in the data."""
        return self._lift("median")

    def mode(self):
        """Return the mode(s) of the data."""
        return self._lift("mode")

    def std(self):
        """Return the stddev(s) of the data."""
        return self._lift("std")

    def count(self):
        """Return the stddev(s) of the data."""
        return self._lift("count")

    # TODO should add reduce here as well...

    @trace
    def agg(self, arg):
        """
        Apply aggregation(s) to the groups.
        """
        # DataFrame{'a': [1, 1, 2], 'b': [1, 2, 3], 'c': [2, 2, 1]})
        # a.groupby('a').agg('sum') -- applied on rest
        # a.groupby('a').agg(['sum', 'min']) -- both applied on rest
        # a.groupby('a').agg({'b': ['min', 'mean']}) -- applied on
        # TODO
        # a.groupby('a').aggregate( a= me['a'].mean(), b_min =me['b'].min(), b_mean=me['c'].mean()))
        # f1 = lambda x: x.quantile(0.5); f1.__name__ = "q0.5"
        # f2 = lambda x: x.quantile(0.75); f2.__name__ = "q0.75"
        # a.groupby('a').agg([f1, f2])

        res = {}
        for f, c in zip(self._key_fields, self._unzip_group_keys()):
            res[f.name] = c
        for agg_name, field, op in self._normalize_agg_arg(arg):
            res[agg_name] = self._apply1(field, op)
        return self._parent._fromdata(res)

    def aggregate(self, arg):
        """
        Apply aggregation(s) to the groups.
        """
        return self.agg(arg)

    @trace
    def select(self, **kwargs):
        """
        Like select for dataframes, except for groups
        """

        res = {}
        for f, c in zip(self._key_fields, self._unzip_group_keys()):
            res[f.name] = c
        for n, c in kwargs.items():
            res[n] = eval_expression(c, {"me": self})
        return self._parent._fromdata(res)

    def _normalize_agg_arg(self, arg):
        res = []  # triple name, field, op
        if isinstance(arg, str):
            # normalize
            arg = [arg]
        if isinstance(arg, list):
            for op in arg:
                for f in self._item_fields:
                    res.append((f"{f.name}.{op}", f, op))
        elif isinstance(arg, dict):
            for n, ops in arg.items():
                fields = [f for f in self._item_fields if f.name == n]
                if len(fields) == 0:
                    raise ValueError(f"column ({n}) does not exist")
                # TODO handle duplicate columns, if ever...
                assert len(fields) == 1
                if isinstance(ops, str):
                    ops = [ops]
                for op in ops:
                    res.append((f"{n}.{op}", fields[0], op))
        else:
            raise TypeError(f"unexpected arg type ({type(arg).__name__})")
        return res


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.Struct.typecode + "_empty", "std"), DataFrameStd._empty)
ColumnFactory.register((dt.Struct.typecode + "_full", "std"), DataFrameStd._full)

# for now we also just use the same code for the CPU...
ColumnFactory.register((dt.Struct.typecode + "_empty", "cpu"), DataFrameStd._empty)
ColumnFactory.register((dt.Struct.typecode + "_full", "cpu"), DataFrameStd._full)


# ------------------------------------------------------------------------------
# DataFrame var (is here and not in Expression) to break cyclic import dependency


# ------------------------------------------------------------------------------
# Relational operators, still TBD


#         def join(
#             self,
#             other,
#             on=None,
#             how="left",
#             lsuffix="",
#             rsuffix="",
#             sort=False,
#             method="hash",
#         ):
#         """Join columns with other DataFrame on index or on a key column."""


#     def rolling(
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

#
#       all set operations: union, uniondistinct, except, etc.
