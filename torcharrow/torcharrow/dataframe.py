#!/usr/bin/env python3
from __future__ import annotations

import array as ar
import copy
import operator
import functools
from collections import OrderedDict
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
    Union,
    cast,
    Tuple,
)

from .column import AbstractColumn, Column, _column_constructor, _set_column_constructor
from .dtypes import (
    CLOSE,
    NL,
    OPEN,
    DType,
    Field,
    ScalarTypes,
    ScalarTypeValues,
    Struct,
    infer_dtype_from_prefix,
    int64,
    is_numerical,
    is_struct,
    is_tuple,
    string,
    Tuple_,
    get_agg_op,
)
from .tabulate import tabulate

from .expression import Var, expression, eval_expression

from .trace import trace, traceproperty
from . import pytorch

# assumes that these have been importd already:
# from .numerical_column import NumericalColumn
# from .string_column import StringColumn
# from .map_column import MapColumn
# from .list_column import MapColumn

# -----------------------------------------------------------------------------
# DataFrames aka (StructColumns, can be nested as StructColumns:-)

DataOrDTypeOrNone = Union[Mapping, Sequence, DType, Literal[None]]


class DataFrame(AbstractColumn):
    """ Dataframe, ordered dict of typed columns of the same length  """

    @trace
    @expression
    def __init__(
        self,
        data: DataOrDTypeOrNone = None,
        dtype: Optional[DType] = None,
        columns: Optional[List[str]] = None,
    ):
        super().__init__(dtype)
        self._field_data = {}

        if data is None and dtype is None:
            assert columns is None
            self._dtype = Struct([])
            return

        if data is not None and isinstance(data, DType):
            if dtype is not None and isinstance(dtype, DType):
                raise TypeError("Dataframe can only have one dtype parameter")
            dtype = data
            data = None

        # dtype given, optional data
        if dtype is not None:
            if not is_struct(dtype):
                raise TypeError(
                    f"Dataframe takes a Struct dtype as parameter (got {dtype})"
                )
            dtype = cast(Struct, dtype)
            self._dtype = dtype
            for f in dtype.fields:
                self._field_data[f.name] = _column_constructor(f.dtype)
            if data is not None:
                if isinstance(data, Sequence):
                    for i in data:
                        self.append(i)
                    return
                elif isinstance(data, Mapping):
                    # start forom scratch
                    self._field_data = {}
                    self._dtype = Struct([])
                    dtype_fields = {f.name: f.dtype for f in dtype.fields}
                    for n, c in data.items():
                        if n not in dtype_fields:
                            raise AttributeError(
                                f"Column {n} is present in the data but absent in explicitly provided dtype"
                            )
                        if isinstance(c, AbstractColumn):
                            if c.dtype != dtype_fields[n]:
                                raise TypeError(
                                    f"Wrong type for column {n}: dtype specifies {dtype_fields[n]} while column of {c.dtype} is provided"
                                )
                        else:
                            c = Column(c, dtype_fields[n])
                        self[n] = c
                        del dtype_fields[n]
                    if len(dtype_fields) > 0:
                        raise TypeError(
                            f"Columns {dtype_fields.keys()} are present in dtype but not provided"
                        )
                    return
                else:
                    raise TypeError(
                        f"Dataframe does not support constructor for data of type {type(data).__name__}"
                    )

        # data given, optional column
        if data is not None:
            if isinstance(data, list):
                prefix = []
                for i, v in enumerate(data):
                    prefix.append(v)
                    if i > 5:
                        break
                dtype = infer_dtype_from_prefix(prefix)
                if dtype is None or not is_tuple(dtype):
                    raise TypeError("Dataframe cannot infer struct type from data")
                dtype = cast(Tuple_, dtype)
                columns = [] if columns is None else columns
                if len(dtype.fields) != len(columns):
                    raise TypeError("Dataframe column length must equal row length")
                self._dtype = Struct(
                    [Field(n, t) for n, t in zip(columns, dtype.fields)]
                )
                for f in self._dtype.fields:
                    self._field_data[f.name] = _column_constructor(f.dtype)
                for i in data:
                    self.append(i)
                return
            elif isinstance(data, dict):
                self._dtype = Struct([])
                for n, c in data.items():
                    self[n] = c if isinstance(c, AbstractColumn) else Column(c)
                return
            else:
                raise TypeError(
                    f"Dataframe does not support constructor for data of type {type(data).__name__}"
                )

    # implementing abstract methods ----------------------------------------------

    @property  # type: ignore
    @traceproperty
    def is_appendable(self):
        """Can this column/frame be extended"""
        return all(
            c.is_appendable and len(c) == self._offset + self._length
            for c in self._field_data.values()
        )

    @trace
    def memory_usage(self, deep=False):
        """Return the memory usage of the Frame (if deep then buffer sizes)."""
        vsize = self._validity.itemsize
        fusage = sum(c.memory_usage(deep) for c in self._field_data.values())
        if not deep:
            return self._length * vsize + fusage
        else:
            return len(self._validity) * vsize + fusage

    @property  # type: ignore
    @traceproperty
    def ndim(self):
        """Column ndim is always 1, Frame ndim is always 2"""
        return 2

    @property  # type: ignore
    @traceproperty
    def size(self):
        """ Number of rows if column; number of rows * number of columns if Frame. """
        return self._length * len(self.columns)

    def _append(self, tup):
        """Append value to the end of the column/frame"""
        if tup is None:
            if not self.dtype.nullable:
                raise TypeError(f"a tuple of type {self.dtype} is required, got None")
            self._null_count += 1
            self._validity.append(False)
            for f in self._dtype.fields:
                # TODO Design decision: null for struct value requires null for its fields
                self._field_data[f.name].append(None)
        else:
            # TODO recursive analysis of tuple structure
            if not isinstance(tup, tuple) and len(tup) == len(self.dtype.fields):
                raise TypeError(
                    f"f'a tuple of type {self.dtype} is required, got {tup}')"
                )
            self._validity.append(True)
            for i, v in enumerate(tup):
                self._field_data[self._dtype.fields[i].name].append(v)
        self._length += 1

    @trace
    @expression
    def get(self, i, fill_value):
        """Get ith row from column/frame"""
        if self._null_count == 0:
            return tuple(self._field_data[f.name][i] for f in self._dtype.fields)
        elif not self._valid(i):
            return fill_value
        else:
            return tuple(self._field_data[f.name][i] for f in self._dtype.fields)

    def __iter__(self):
        """Return the iterator object itself."""
        for i in range(self._length):
            j = self._offset + i
            if self._validity[j]:
                yield tuple(self._field_data[f.name][j] for f in self._dtype.fields)
            else:
                yield None

    def _copy(self, deep, offset, length):
        if deep:
            res = DataFrame(self.dtype)
            res._length = length
            # TODO optimize (here and everywhere): only non-appendable columns need to be copied
            res._field_data = {
                n: c._copy(deep, offset, length) for n, c in self._field_data.items()
            }
            res._validity = self._validity[offset : offset + length]
            res._null_count = sum(res._validity)
            return res
        else:
            return copy.copy(self)

    def _raw_lengths(self):
        return AbstractColumn._flatten(
            [c._raw_lengths() for c in self._field_data.values()]
        )

    # implementing abstract methods ----------------------------------------------

    @property  # type: ignore
    @traceproperty
    def columns(self):
        """The column labels of the DataFrame."""
        return list(self._field_data.keys())

    @trace
    def __setitem__(self, name: str, value: Any) -> None:
        d = None
        if isinstance(value, AbstractColumn):
            d = value
        elif isinstance(value, Iterable):
            d = Column(value)
        else:
            raise TypeError("data must be a column or list")

        assert d is not None
        if all(len(c) == 0 for c in self._field_data.values()):
            self._length = len(d)
            self._validity = ar.array("b", [True] * self._length)

        elif len(d) != self._length:
            raise TypeError("all columns/lists must have equal length")

        if name in self._field_data.keys():
            raise AttributeError(f"cannot override existing column {name}")
        elif (
            self._dtype is not None
            and isinstance(self._dtype, Struct)
            and len(self._dtype.fields) < len(self._field_data)
        ):
            raise AttributeError("cannot append column to view")
        else:
            assert self._dtype is not None and isinstance(self._dtype, Struct)
            # side effect on field_data
            self._field_data[name] = d
            # no side effect on dtype
            fields = list(self._dtype.fields)
            assert d._dtype is not None
            fields.append(Field(name, d._dtype))
            self._dtype = Struct(fields)

    def to_python(self):
        tup_type = self._dtype.py_type
        # TODO: we probably don't need subscript here with offset after df[1:3]["A"] slicing is fixed
        return [
            tup_type(*v)
            for v in zip(
                *(
                    self._field_data[f.name][
                        self._offset : self._offset + self._length
                    ].to_python()
                    for f in self._dtype.fields
                )
            )
        ]

    def to_torch(self):
        pytorch.ensure_available()
        import torch

        # TODO: this actually puts the type annotations on the tuple wrong. We might need to address it eventually, but because it's python it doesn't matter
        tup_type = self._dtype.py_type
        # TODO: we probably don't need subscript here with offset after df[1:3]["A"] slicing is fixed
        for f in self._dtype.fields:
            self._field_data[f.name][
                self._offset : self._offset + self._length
            ].to_torch()
        return tup_type(
            *(
                self._field_data[f.name][
                    self._offset : self._offset + self._length
                ].to_torch()
                for f in self._dtype.fields
            )
        )

    # printing ----------------------------------------------------------------
    def __str__(self):
        def quote(n):
            return f"'{n}'"

        return f"DataFrame({OPEN}{', '.join(f'{quote(n)}:{str(c)}' for n,c in self._field_data.items())}, id = {self.id}{CLOSE})"

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
        typ = f"dtype: {self._dtype}, count: {self._length}, null_count: {self._null_count}"
        return tab + NL + typ

    def show_details(self):
        return _Repr(self)

    # selectors -----------------------------------------------------------
    def _get_column(self, arg):
        return self._field_data[arg]

    def _index(self, arg):
        if arg is None:
            return None
        if arg in self.columns:
            return self.columns.index(arg)
        try:
            return int(arg)
        except:
            raise TypeError(
                "column slice indices must be column names or strings that can be converted to integers or None"
            )

    def _slice_columns(self, arg):
        columns = self.columns

        start_ = self._index(arg.start)
        stop_ = self._index(arg.stop)
        step_ = None
        if arg.step is not None:
            try:
                step_ = int(arg.step)
            except:
                raise TypeError(
                    "column step argument must a string that can be converted to an integer or None"
                )

        start, stop, step = slice(start_, stop_, step_).indices(len(columns))

        res = DataFrame()
        for i in range(start, stop, step):
            res[columns[i]] = self._field_data[columns[i]]
        return res

    def _pick_columns(self, arg):
        res = DataFrame()
        for i in arg:
            res[i] = self._field_data[i]
        return res

    # functools map/filter/reduce ---------------------------------------------
    @trace
    @expression
    def map(
        self,
        arg: Union[Dict, Callable],
        /,
        na_action: Literal["ignore", None] = None,
        dtype: Optional[DType] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Maps rows according to input correspondence.
        dtype required if result type != item type.
        """
        if columns is None:
            return super().map(arg, na_action, dtype)
        for i in columns:
            if i not in self.columns:
                raise KeyError("column {i} not in dataframe")
        if len(columns) == 1:
            return self._map_unary(arg, na_action, dtype, columns[0])
        else:
            return self._map_nary(arg, na_action, dtype, columns)

    def _map_unary(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None],
        dtype: Optional[DType],
        column: str,
    ):
        func, dtype = self._normalize_map_arg(arg, dtype)

        res = _column_constructor(dtype)
        for i in range(self._length):
            if self._valid(i) or na_action == "ignore":
                res._append(func(self._field_data[column][i]))
            else:
                res._append(None)
        return res

    def _map_nary(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None],
        dtype: Optional[DType],
        columns: List[str],
    ):
        if isinstance(arg, dict):
            # the rule for nary map is different!
            new_arg = lambda *x: arg.get(tuple(*x), None)
            arg = new_arg

        func, dtype = self._normalize_map_arg(arg, dtype)

        res = _column_constructor(dtype)
        for i in range(self._length):
            if self._valid(i) or na_action == "ignore":
                res._append(func(*[self._field_data[n][i] for n in columns]))
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
        if columns is None:
            return super().flatmap(arg, na_action, dtype)
        for i in columns:
            if i not in self.columns:
                raise KeyError("column {i} not in dataframe")
        if len(columns) == 1:
            return self._unary_flatmap(arg, na_action, dtype, columns[0])
        else:
            return self._nary_flatmap(arg, na_action, dtype, columns)

    def _unary_flatmap(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None],
        dtype: Optional[DType],
        column: str,
    ):
        def func(x):
            return arg.get(x, None) if isinstance(arg, dict) else arg(x)

        dtype1 = dtype if dtype is not None else self._dtype
        res = _column_constructor(dtype1)
        for i in range(self._length):
            if self._valid(i) or na_action == "ignore":
                res.extend(func(self._field_data[column][i]))
            else:
                res._append(None)
        return res

    def _nary_flatmap(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None],
        dtype: Optional[DType],
        columns: List[str],
    ):
        def func(x):
            return arg.get(x, None) if isinstance(arg, dict) else arg(x)

        dtype1 = dtype if dtype is not None else self._dtype
        res = _column_constructor(dtype1)
        for i in range(self._length):
            if self._valid(i) or na_action == "ignore":
                res.extend(func(*[self._field_data[n][i] for n in columns]))
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
        if columns is None:
            return super().filter(predicate)
        for c in columns:
            if c not in self.columns:
                raise KeyError("column {c} not in dataframe")

        if not isinstance(predicate, Iterable) and not callable(predicate):
            raise TypeError(
                "predicate must be a unary boolean predicate or iterable of booleans"
            )
        res = DataFrame(self._dtype)
        if callable(predicate):
            for i in range(self._length):
                if predicate(*[self._field_data[n][i] for n in columns]):
                    res._append(self[i])
        elif isinstance(predicate, Iterable):
            for x, p in zip(self, predicate):
                if p:
                    res._append(x)
        else:
            pass
        return res

    # ifthenelse -----------------------------------------------------------------

    # def where(self, cond, other):
    #     """Replace values where the condition is False; other must have same type and size as self."""
    #     res = DataFrame()
    #     for (n, c), (m, d) in zip(self._field_data.items(), other._field_data.items()):
    #         if (n != m) or (len(c) != len(d)):
    #             # TODO add type chcek
    #             raise TypeError(
    #                 f"column names,types and lengths have to match between self['{n}] and other['{m}']")
    #     for n, c in self._field_data.items():
    #         d = other._field_data[n]
    #         res[n] = c.where(cond, d)
    #     return res

    # def applymap(self, func, na_action: Literal['ignore', None] = None, dtype: Optional[DType] = None):
    #     """Apply a function to a Dataframe elementwise"""
    #     return self._lift(lambda c: c._map, func=func, na_action=na_action, dtype=dtype)

    # sorting ----------------------------------------------------------------

    @trace
    @expression
    def sort_values(
        self,
        by: Union[str, List[str], Literal[None]] = None,
        ascending=True,
        na_position: Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order"""
        # Not allowing None in comparison might be too harsh...
        # Move all rows with None that in sort index to back...
        func = None
        if isinstance(by, str):
            by = [by]
        if isinstance(by, list):
            xs = []
            for i in by:
                _ = self._field_data[i]  # throws key error
                xs.append(self.columns.index(i))
            reorder = xs + [j for j in range(len(self._field_data)) if j not in xs]

            def func(tup):
                return tuple(tup[i] for i in reorder)

        res = DataFrame(self.dtype)
        if na_position == "first":
            res.extend([None] * self._null_count)
        res.extend(
            sorted((i for i in self if i is not None), reverse=not ascending, key=func)
        )
        if na_position == "last":
            res.extend([None] * self._null_count)
        return res

    @trace
    @expression
    def sort(
        self,
        by: Union[str, List[str], Literal[None]] = None,
        ascending=True,
        na_position: Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order (aka sort_values)"""
        return self.sort_values(by, ascending, na_position)

    @trace
    @expression
    def nlargest(
        self,
        n=5,
        columns: Union[str, List[str], Literal[None]] = None,
        keep: Literal["last", "first"] = "first",
    ):
        """Returns a new dataframe of the *n* largest elements."""
        # Todo add keep arg
        return self.sort_values(by=columns, ascending=False).head(n)

    @trace
    @expression
    def nsmallest(
        self,
        n=5,
        columns: Union[str, List[str], Literal[None]] = None,
        keep: Literal["last", "first"] = "first",
    ):
        """Returns a new dataframe of the *n* smallest elements. """
        # keep="all" not supported
        # Todo add keep arg
        return self.sort_values(by=columns, ascending=True).head(n)

    # operators --------------------------------------------------------------

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
        return self._binary_operator(
            "floordiv",
            other,
            fill_value=fill_value,
        )

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

    def le(self, other, fill_value=None):
        return self._binary_operator("le", other, fill_value=fill_value)

    def __le__(self, other):
        return self._binary_operator("le", other)

    def ge(self, other, fill_value=None):
        return self._binary_operator("ge", other, fill_value=fill_value)

    def __ge__(self, other):
        return self._binary_operator("ge", other)

    # bitwise (reused for logical)

    # bitwise or (|), used for logical or
    def __or__(self, other):
        return self._binary_operator("or", other)

    def __ror__(self, other):
        return self._binary_operator("or", other, reflect=True)

    # bitwise and (&), used for logical and
    def __and__(self, other):
        return self._binary_operator("and", other)

    def __rand__(self, other):
        return self._binary_operator("and", other, reflect=True)

    # bitwise complement (~), used for logical not
    def __invert__(self):
        return self._unary_operator(operator.inv)

    # arithmetic
    def __neg__(self):
        return self._unary_operator(operator.neg)

    def __pos__(self):
        return self._unary_operator(operator.pos)

    @trace
    def _lift(self, func, /, kwargs):
        res = DataFrame()
        if self._null_count == 0:
            for n, c in self._field_data.items():
                res[n] = func(c)(**kwargs)
            return res
        raise NotImplementedError("Dataframe row is not allowed to have nulls")

    def _lift_pairs(self, func, other, /, kwargs):
        res = DataFrame()
        if self._null_count == 0:
            for n, c in self._field_data.items():
                res[n] = func(c)(**{"other": other[n], **kwargs})
            return res
        raise NotImplementedError("Dataframe row is not allowed to have nulls")

    @trace
    def _unary_operator(self, operator):
        res = DataFrame()
        if self._null_count == 0:
            for n, c in self._field_data.items():
                res[n] = c._unary_operator(operator, c.dtype)
            return res
        raise NotImplementedError("Dataframe row is not allowed to have nulls")

    @trace
    def _binary_operator(self, operator, other, fill_value=None, reflect=False):

        if isinstance(other, ScalarTypeValues):
            return self._lift(
                lambda c: c._binary_operator,
                {
                    "operator": operator,
                    "other": other,
                    "fill_value": fill_value,
                    "reflect": reflect,
                },
            )
        elif isinstance(other, DataFrame):  # order important
            return self._lift_pairs(
                lambda c: c._binary_operator,
                other,
                {"operator": operator, "fill_value": fill_value, "reflect": reflect},
            )
        elif isinstance(other, AbstractColumn):  # order important
            # replicate column to match Dataframe:
            other_replicated = DataFrame()
            for n in self._field_data.keys():
                other_replicated[n] = other
            return self._lift_pairs(
                lambda c: c._binary_operator,
                other_replicated,
                {"operator": operator, "fill_value": fill_value, "reflect": reflect},
            )

        else:
            raise TypeError(
                f"cannot apply '{operator}' on {type(self).__name__} and {type(other).__name__}"
            )

    # isin ---------------------------------------------------------------

    @trace
    @expression
    def isin(self, values: Union[list, dict, AbstractColumn]):
        """Check whether values are contained in data."""
        res = DataFrame()
        if isinstance(values, Iterable):
            return self._lift(lambda c: c.isin, {"values": values})
        if isinstance(values, dict):
            for i in values.keys():
                _ = self[i]  # throws key error
            res = DataFrame()
            for n, c in self._field_data.items():
                res[n] = c.isin(values=values[n])  # throws key error
            return res
        if isinstance(values, DataFrame):
            for i in values.keys():
                _ = self[i]  # throws key error
            res = DataFrame()
            for n, c in self._field_data.items():
                res[n] = c.isin(values=list(values[n]))  # throws key error
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
        if fill_value is None:
            return self

        if isinstance(fill_value, ScalarTypeValues):
            return self._lift(lambda c: c.fillna, {"fill_value": fill_value})
        elif isinstance(fill_value, dict):
            res = DataFrame()
            # Dead code?
            # for n in fill_value.keys():
            #     _ = self._field_data[n]  # throw key error for undefined keys
            # for n, c in self._field_data.items():
            #     res[n] = c.fillna(fill_value=fill_value[n]) if n in dict else c
            return res
        elif isinstance(fill_value, DataFrame):
            res = DataFrame()
            # Dead code?
            # if self._shapeix != fill_value._shapeix:
            #     TypeError(
            #         f"fillna between differently 'shaped' and 'indexed' dataframes is not supported"
            #     )

            # "DataFrame" has no attribute "values"
            # for (n, c), d in zip(self._field_data.items(), fill_value.values()):
            #     res[n] = c.fillna(fill_value=d)
            return res
        else:
            raise TypeError(f"fillna with {type(fill_value)} is not supported")

    @trace
    @expression
    def dropna(self, how: Literal["any", "all"] = "any"):
        """Return a dataframe with rows removed where the row has any or all nulls."""
        # TODO only flat columns supported...
        assert self._dtype is not None
        res = DataFrame(self._dtype.constructor(nullable=False))
        if how == "any":
            for i in self:
                if not DataFrame._has_any_null(i):
                    res._append(i)
        elif how == "all":
            for i in self:
                if not DataFrame._has_all_null(i):
                    res._append(i)
        return res

    @trace
    @expression
    def drop_duplicates(
        self,
        subset: Union[str, List[str], Literal[None]] = None,
        keep: Literal["first", "last", False] = "first",
    ):
        """ Remove duplicate values from data but keep the first, last, none (keep=False)"""
        # Todo Add functionality
        assert keep == "first"
        res = DataFrame(self.dtype)
        if subset is None:
            res.extend(list(OrderedDict.fromkeys(self)))
        else:
            if isinstance(subset, str):
                subset = [subset]

            if isinstance(subset, list):
                for s in subset:
                    _ = self._field_data[s]  # throws key error
                idxs = [self.columns.index(s) for s in subset]
                seen = set()
                for tup in self:
                    row = tuple(tup[i] for i in idxs)
                    if row in seen:
                        continue
                    else:
                        seen.add(row)
                        res._append(tup)
        return res

    @staticmethod
    def _has_any_null(tup) -> bool:
        for t in tup:
            if t is None:
                return True
            if isinstance(t, tuple) and DataFrame._has_any_null(t):
                return True
        return False

    @staticmethod
    def _has_all_null(tup) -> bool:
        for t in tup:
            if t is not None:
                return False
            if isinstance(t, tuple) and not DataFrame._has_all_null(t):
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
    def min(self, numeric_only=None):
        """Return the minimum of the nonnull values of the Column."""
        return self._summarize(DataFrame._cmin, {"numeric_only": numeric_only})

    # with dataclass function
    # @expression
    # def min(self, numeric_only=None):
    #     """Return the minimum of the nonnull values of the Column."""
    #     return self._summarize(_Min(), {"numeric_only": numeric_only})

    # with lambda
    # @expression
    # def min(self, numeric_only=None):
    #     """Return the minimum of the nonnull values of the Column."""
    #     return self._summarize(lambda c: c.min, {"numeric_only": numeric_only})

    @trace
    @expression
    def max(self, numeric_only=None):
        """Return the maximum of the nonnull values of the column."""
        # skipna == True
        return self._summarize(lambda c: c.max, {"numeric_only": numeric_only})

    @trace
    @expression
    def all(self, boolean_only=None):
        """Return whether all nonull elements are True in Column"""
        return self._summarize(lambda c: c.all, {"boolean_only": boolean_only})

    @trace
    @expression
    def any(self, skipna=True, boolean_only=None):
        """Return whether any nonull element is True in Column"""
        return self._summarize(lambda c: c.any, {"boolean_only": boolean_only})

    @trace
    @expression
    def sum(self):
        """Return sum of all nonull elements in Column"""
        return self._summarize(lambda c: c.sum, {})

    @trace
    @expression
    def prod(self):
        """Return produce of the values in the data"""
        return self._summarize(lambda c: c.prod, {})

    @trace
    @expression
    def cummin(self, skipna=True):
        """Return cumulative minimum of the data."""
        return self._lift(lambda c: c.cummin, {"skipna": skipna})

    @trace
    @expression
    def cummax(self, skipna=True):
        """Return cumulative maximum of the data."""
        return self._lift(lambda c: c.cummax, {"skipna": skipna})

    @trace
    @expression
    def cumsum(self, skipna=True):
        """Return cumulative sum of the data."""
        return self._lift(lambda c: c.cumsum, {"skipna": skipna})

    @trace
    @expression
    def cumprod(self, skipna=True):
        """Return cumulative product of the data."""
        return self._lift(lambda c: c.cumprod, {"skipna": skipna})

    @trace
    @expression
    def mean(self):
        """Return the mean of the values in the series."""
        return self._summarize(lambda c: c.mean, {})

    @trace
    @expression
    def median(self):
        """Return the median of the values in the data."""
        return self._summarize(lambda c: c.median, {})

    @trace
    @expression
    def mode(self):
        """Return the mode(s) of the data."""
        return self._summarize(lambda c: c.mode, {})

    @trace
    @expression
    def std(self):
        """Return the stddev(s) of the data."""
        return self._summarize(lambda c: c.std, {})

    @trace
    @expression
    def nunique(self, dropna=True):
        """Returns the number of unique values per column"""
        res = DataFrame(Struct([Field("column", string), Field("nunique", int64)]))
        for n, c in self._field_data.items():
            res._append((n, c.nunique(dropna)))
        return res

    def _summarize(self, func, /, kwargs):
        res = DataFrame()
        # if self._null_count == 0:
        for n, c in self._field_data.items():
            res[n] = Column([func(c)(**kwargs)])
        return res
        # raise NotImplementedError('Dataframe row is not allowed to have nulls')

    # describe ----------------------------------------------------------------

    @trace
    @expression
    def describe(
        self,
        percentiles=None,
        include_columns=None,
        exclude_columns=None,
    ):
        """Generate descriptive statistics."""
        # Not supported: datetime_is_numeric=False,
        includes = []
        if include_columns is None:
            includes = [n for n, c in self._field_data.items() if is_numerical(c.dtype)]
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

        res = DataFrame()
        res["metric"] = (
            ["count", "mean", "std", "min"] + [f"{p}%" for p in percentiles] + ["max"]
        )
        for s in selected:
            c = self._field_data[s]
            res[s] = (
                [c.count(), c.mean(), c.std(), c.min()]
                + c._percentiles(percentiles)
                + [c.max()]
            )
        return res

    # Dataframe specific ops --------------------------------------------------    #

    @trace
    @expression
    def drop(self, columns: List[str]):
        """
        Returns DataFrame without the removed columns.
        """
        if isinstance(columns, list):
            for n in columns:
                _ = self._field_data[n]  # creates key error
            res = DataFrame()
            for n, c in self._field_data.items():
                if n not in columns:
                    res[n] = c
            return res
        else:
            raise TypeError(
                f"drop with column parameter of type {type(columns).__name__} is not supported"
            )

    @trace
    @expression
    def keep(self, columns: List[str]):
        """
        Returns DataFrame with the kept columns only.
        """
        if isinstance(columns, list):
            for n in columns:
                _ = self._field_data[n]  # creates key error
            res = DataFrame()
            for n, c in self._field_data.items():
                if n in columns:
                    res[n] = c
            return res
        else:
            raise TypeError(
                f"keep with column parameter of type {type(columns).__name__} is not supported"
            )

    @trace
    @expression
    def rename(self, column_mapper: Dict[str, str]):
        if isinstance(column_mapper, dict):
            for n in column_mapper:
                _ = self._field_data[n]  # creates key error
            res = DataFrame()
            for n, c in self._field_data.items():
                if n in column_mapper:
                    res[column_mapper[n]] = c
                else:
                    res[n] = c
            return res
        else:
            raise TypeError(
                f"rename with column_mapper parameter of type {type(column_mapper).__name__} is not supported"
            )

    @trace
    @expression
    def reorder(self, columns: List[str]):
        if isinstance(columns, list):
            for n in columns:
                _ = self._field_data[n]  # creates key error
        res = DataFrame()
        for n in columns:
            res[n] = self._field_data[n]
        return res

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

        if not conditions:
            return self

        evalled_conditions = [
            eval_expression(condition, {"me": self}) for condition in conditions
        ]
        anded_evalled_conditions = functools.reduce(
            lambda x, y: x & y, evalled_conditions
        )
        return self[anded_evalled_conditions]

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

        res = DataFrame()
        for n, c in self._field_data.items():
            if n in output_columns:
                res[n] = c
        for n, c in kwargs.items():
            res[n] = eval_expression(c, {"me": self})
        return res

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
        self,
        by: List[str],
        sort=False,
        dropna=True,
    ):
        # TODO implement
        assert not sort
        assert dropna

        key_columns = by
        key_fields = []
        item_fields = []
        for k in key_columns:
            if k not in self._field_data.keys():
                raise ValueError(f"groupby received a non existent column ({k})")
            key_fields.append(Field(k, self.dtype.get(k)))
        for f in self.dtype.fields:
            if f.name not in key_columns:
                item_fields.append(f)

        groups: Dict[Tuple, ar.array] = {}
        for i in range(self._length):
            j = self._offset + i
            if self._validity[j]:
                key = tuple(self._field_data[f.name][j] for f in key_fields)
                if key not in groups:
                    groups[key] = ar.array("I")
                df = groups[key]
                df.append(j)
            else:
                pass
        return GroupedDataFrame(key_fields, item_fields, groups, self)


@dataclass
class GroupedDataFrame:
    _key_fields: List[Field]
    _item_fields: List[Field]
    _groups: Mapping[Tuple, Sequence]
    _parent: DataFrame

    @property  # type: ignore
    @traceproperty
    def size(self):
        """
        Return the size of each group (including nulls).
        """
        res = DataFrame(Struct(self._key_fields + [Field("size", int64)]))
        for k, c in self._groups.items():
            row = k + (len(c),)
            res._append(row)
        return res

    def __iter__(self):
        """
        Yield pairs of grouped tuple and the grouped dataframe
        """
        for g, xs in self._groups.items():
            df = DataFrame(Struct(self._item_fields))
            for x in xs:
                df.append(
                    tuple(
                        self._parent._field_data[f.name][x] for f in self._item_fields
                    )
                )
            yield g, df

    @trace
    def _lift(self, op: str) -> AbstractColumn:
        if len(self._key_fields) > 0:
            # it is a dataframe operation:
            return self._combine(op)
        elif len(self._item_fields) == 1:
            return self._apply1(self._item_fields[0], op)
        raise AssertionError("unexpected case")

    def _combine(self, op: str) -> DataFrame:
        agg_fields = [Field(f"{f.name}.{op}", f.dtype) for f in self._item_fields]
        res = DataFrame()
        for f, c in zip(self._key_fields, self._unzip_group_keys()):
            res[f.name] = c
        for f, c in zip(agg_fields, self._apply(op)):
            res[f.name] = c
        return res

    def _apply(self, op: str) -> List[AbstractColumn]:
        cols = []
        for f in self._item_fields:
            cols.append(self._apply1(f, op))
        return cols

    def _apply1(self, f: Field, op: str) -> AbstractColumn:
        src_t = f.dtype
        dest_f, dest_t = get_agg_op(op, src_t)
        col = _column_constructor(dest_t)
        src_c = self._parent._field_data[f.name]
        for g, xs in self._groups.items():
            dest_c = dest_f(Column((src_c[x] for x in xs), dtype=dest_t))
            col.append(dest_c)
        return col

    def _unzip_group_keys(self) -> List[AbstractColumn]:
        cols = []
        for f in self._key_fields:
            cols.append(_column_constructor(f.dtype))
        for tup in self._groups.keys():
            for i, t in enumerate(tup):
                cols[i].append(t)
        return cols

        # res_fields = []
        # tmp_ops = []
        # for f in self._item_fields:
        #     c_op, c_type = get_agg_op(op, f.dtype)
        #     res_fields.append(Field(f"{f.name}.{op}", c_type))
        #     tmp_ops.append(c_op)

        # res = DataFrame(self._key_fields + self._res_fields)
        # for g, xs in self._groups.items():
        #     tup = list(g)
        #     for i, (f, r) in enumerate(zip(self._item_fields, res_fields)):
        #         tup.append(
        #             tmp_ops[i](Column((self._parent._field_data[f.name][x]
        #                        for x in xs), dtype=r.dtype)))
        #     res.append(tuple(tup))
        # return res

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
                    col = _column_constructor(f.dtype)
                    for tup in self._groups.keys():
                        col.append(tup[i])
                    return col
            raise ValueError(f"no column named ({arg}) in grouped dataframe")
        raise TypeError(f"unexpected type for arg ({type(arg).__name})")

    def min(self, numeric_only=None):
        """Return the minimum of the nonnull values of the Column."""
        assert numeric_only == None
        return self._lift("min")

    def max(self, numeric_only=None):
        """Return the minimum of the nonnull values of the Column."""
        assert numeric_only == None
        return self._lift("min")

    def all(self, boolean_only=None):
        """Return whether all nonull elements are True in Column"""
        # skipna == True
        return self._lift("all")

    def any(self, skipna=True, boolean_only=None):
        """Return whether any nonull element is True in Column"""
        # skipna == True
        return self._lift("any")

    def sum(self):
        """Return sum of all nonull elements in Column"""
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

        res = DataFrame()
        for f, c in zip(self._key_fields, self._unzip_group_keys()):
            res[f.name] = c
        for agg_name, field, op in self._normalize_agg_arg(arg):
            res[agg_name] = self._apply1(field, op)
        return res

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

        res = DataFrame()
        for f, c in zip(self._key_fields, self._unzip_group_keys()):
            res[f.name] = c
        for n, c in kwargs.items():
            res[n] = eval_expression(c, {"me": self})
        return res

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

        # (lambda c: c.sum())(df['b'])
        # TODO
        # def pipe(self, func, *args, **kwargs):
        #     """
        #     Apply func(self, \*args, \*\*kwargs).
        #     """
        #     df = DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': [1, 2, 3, 4]})
        #     df.groupby('A').pipe({'B': lambda x: x.max() - x.min()}


@dataclass
class _Repr:
    parent: DataFrame

    def __repr__(self):
        raise NotImplementedError()


# ------------------------------------------------------------------------------
# registering the factory
_set_column_constructor(is_struct, DataFrame)
_set_column_constructor(is_tuple, DataFrame)

# ------------------------------------------------------------------------------
# DataFrame var (is here and not in Expression) to break cyclic import depedency


class DataFrameVar(Var, DataFrame):
    def __init__(self, name: str):
        super().__init__(name)


# The super variable...
me = DataFrameVar("me")


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

#     def apply_rows(
#         self,
#         func,
#         incols,
#         outcols,
#         kwargs,
#         pessimistic_nulls=True,
#         cache_key=None,
#     ):
#         """
#         Apply a row-wise user defined function.
#
#       all set operations: union, uniondistinct, except, etc.
