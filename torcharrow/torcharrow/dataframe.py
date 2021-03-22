import array as ar
import copy
import operator
import functools
from collections import OrderedDict
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
                    Optional, Sequence, Union)

from .column import (AbstractColumn, Column, _column_constructor,
                     _set_column_constructor)
from .dtypes import (CLOSE, NL, OPEN, DType, Field, ScalarTypes,
                     ScalarTypeValues, Struct, infer_dtype_from_prefix, int64,
                     is_numerical, is_struct, is_tuple, string)
from .tabulate import tabulate


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

    def __init__(self, data: DataOrDTypeOrNone = None, dtype: Optional[DType] = None, columns: Optional[List[str]] = None):

        super().__init__(dtype)
        self._field_data = {}

        if data is None and dtype is None:
            assert columns is None
            self._dtype = Struct([])
            return

        if data is not None and isinstance(data, DType):
            if dtype is not None and isinstance(dtype, DType):
                raise TypeError('Dataframe can only have one dtype parameter')
            dtype = data
            data = None

        # dtype given, optional data
        if dtype is not None:
            if not is_struct(dtype):
                raise TypeError(
                    f'Dataframe takes a Struct dtype as parameter (got {dtype})')
            self._dtype = dtype
            for f in dtype.fields:
                self._field_data[f.name] = _column_constructor(f.dtype)
            if data is not None:
                if isinstance(data, Sequence):
                    for i in data:
                        self.append(i)
                    return
                elif isinstance(data, Mapping):
                    for n, c in data.items():
                        self[n] = c if isinstance(
                            c, AbstractColumn) else Column(c)
                    return
                else:
                    raise TypeError(
                        f'Dataframe does not support constructor for data of type {type(data).__name__}')

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
                    raise TypeError(
                        'Dataframe cannot infer struct type from data')
                columns = [] if columns is None else columns
                if len(dtype.fields) != len(columns):
                    raise TypeError(
                        'Dataframe column length must equal row length')
                self._dtype = Struct([Field(n, t)
                                     for n, t in zip(columns, dtype.fields)])
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
                    f'Dataframe does not support constructor for data of type {type(data).__name__}')

    # implementing abstract methods ----------------------------------------------

    @property
    def is_appendable(self):
        """Can this column/frame be extended"""
        return all(c.is_appendable and len(c) == self._offset + self._length for c in self._field_data.values())

    def memory_usage(self, deep=False):
        """Return the memory usage of the Frame (if deep then buffer sizes)."""
        vsize = self._validity.itemsize
        fusage = sum(c.memory_usage(deep) for c in self._field_data.values())
        if not deep:
            return self._length * vsize + fusage
        else:
            return len(self._validity)*vsize + fusage

    @property
    def ndim(self):
        """Column ndim is always 1, Frame ndim is always 2"""
        return 2

    @property
    def size(self):
        """ Number of rows if column; number of rows * number of columns if Frame. """
        return self._length * len(self.columns)

    def _append(self, tup):
        """Append value to the end of the column/frame"""
        if tup is None:
            if not self.dtype.nullable:
                raise TypeError(
                    f'a tuple of type {self.dtype} is required, got None')
            self._null_count += 1
            self._validity.append(False)
            for f in self._dtype.fields:
                # TODO Design decision: null for struct value requires null for its fields
                self._field_data[f.name].append(None)
        else:
            # TODO recursive analysis of tuple structure
            if not isinstance(tup, tuple) and len(tup) == len(self.dtype.files):
                raise TypeError(
                    f"f'a tuple of type {self.dtype} is required, got {tup}')")
            self._validity.append(True)
            for i, v in enumerate(tup):
                self._field_data[self._dtype.fields[i].name].append(v)
        self._length += 1

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
            res._field_data = {n: c._copy(deep, offset, length)
                               for n, c in self._field_data.items()}
            res._validity = self._validity[offset: offset+length]
            res._null_count = sum(res._validity)
            return res
        else:
            return copy.copy(self)

    def _raw_lengths(self):
        return AbstractColumn._flatten([c._raw_lengths() for c in self._field_data.values()])

    # implementing abstract methods ----------------------------------------------

    @property
    def columns(self):
        """The column labels of the DataFrame."""
        return list(self._field_data.keys())

    def __setitem__(self, name: str, value: Any) -> None:
        d = None
        if isinstance(value, AbstractColumn):
            d = value
        elif isinstance(value, Iterable):
            d = Column(value)
        else:
            raise TypeError('data must be a column or list')

        if all(len(c) == 0 for c in self._field_data.values()):
            self._length = len(d)
            self._validity = ar.array('b', [True] * self._length)

        elif len(d) != self._length:
            raise TypeError('all columns/lists must have equal length')

        if name in self._field_data.keys():
            raise AttributeError('cannot override existing column')
        elif len(self._dtype.fields) < len(self._field_data):
            raise AttributeError('cannot append column to view')
        else:
            # side effect on field_data
            self._field_data[name] = d
            # no side effect on dtype
            fields = list(self._dtype.fields)
            fields.append(Field(name, d._dtype))
            self._dtype = Struct(fields)

    # printing ----------------------------------------------------------------
    def __str__(self):
        def quote(n): return f"'{n}'"
        return f"DataFrame({OPEN}{', '.join(f'{quote(n)}:{str(c)}' for n,c in self._field_data.items())}{CLOSE})"

    def __repr__(self):
        data = []
        for i in self:
            if i is None:
                data.append(['None'] * len(self.columns))
            else:
                assert len(i) == len(self.columns)
                data.append(list(i))
        tab = tabulate(
            data, headers=["index"]+self.columns, tablefmt='simple', showindex=True)
        typ = f"dtype: {self._dtype}, count: {self._length}, null_count: {self._null_count}"
        return tab+NL+typ

    def show_details(self):
        return _Repr(self)

    # selectors -----------------------------------------------------------
    def _get_column(self, arg, default=None):
        # TODO delete the default, no?
        return self._field_data[arg]

    def _slice_columns(self, arg):
        if arg.step is not None:
            raise ValueError(
                'slicing column names requires step parameter to be None')

        start = 0
        columns = self.columns
        if arg.start is not None:
            self._field_data[arg.start]  # triggers keyerror
            start = columns.index(arg.start)
        else:
            start = 0
        if arg.stop is not None:
            self._field_data[arg.stop]  # trigger keyerror
            stop = columns.index(arg.stop)
        else:
            stop = len(columns)
        res = DataFrame()
        for i in range(start, stop):
            res[columns[i]] = self._field_data[columns[i]]
        return res

    def _pick_columns(self, arg):
        res = DataFrame()
        for i in arg:
            res[i] = self._field_data[i]
        return res

    # conversions -------------------------------------------------------------

    # map and filter -----------------------------------------------------------
    # TODO: Have to decide on which meaning to give to filter, map, where.
    # Right now map, filter, flatmap simply extend from Column to Dataframe
    # So this is commented out!
    # def map(
    #     self,
    #     arg: Union[Dict, Callable],
    #     na_action: Literal["ignore", None] = None,
    #     dtype: Optional[DType] = None,
    # ):
    #     """
    #     Map rows according to input correspondence.
    #     dtype required if result type != item type.
    #     """
    #     res = DataFrame()
    #     for n, c in self._field_data.items():
    #         res[n] = c.map(arg, na_action, dtype)
    #     return res

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

    def sort_values(
        self,
        by: Union[str, List[str], Literal[None]] = None,
        ascending=True,
        na_position: Literal['last', 'first'] = "last",

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
            reorder = xs + \
                [j for j in range(len(self._field_data)) if j not in xs]

            def func(tup): return tuple(tup[i] for i in reorder)

        res = DataFrame(self.dtype)
        if na_position == 'first':
            res.extend([None] * self._null_count)
        res.extend(sorted((i for i in self if i is not None),
                   reverse=not ascending, key=func))
        if na_position == 'last':
            res.extend([None] * self._null_count)
        return res

    def sort(
        self,
        by: Union[str, List[str], Literal[None]] = None,
        ascending=True,
        na_position: Literal['last', 'first'] = "last",

    ):
        """Sort a column/a dataframe in ascending or descending order (aka sort_values)"""
        return self.sort_values(by, ascending, na_position)

    def nlargest(self,
                 n=5,
                 columns: Union[str, List[str], Literal[None]] = None,
                 keep: Literal['last', 'first'] = "first"):
        """Returns a new data of the *n* largest element."""
        # Todo add keep arg
        return self.sort_values(by=columns, ascending=False).head(n)

    def nsmallest(self,
                  n=5,
                  columns: Union[str, List[str], Literal[None]] = None,
                  keep: Literal['last', 'first'] = "first"):
        """Returns a new data of the *n* smallest element. """
        # keep="all" not supported
        # Todo add keep arg
        return self.sort_values(by=columns, ascending=True).head(n)

    # operators --------------------------------------------------------------

    # arithmetic

    def add(self, other, fill_value=None):
        return self._binary_operator("add", other, fill_value=fill_value)

    def radd(self, other, fill_value=None):
        return self._binary_operator("add", other, fill_value=fill_value, reflect=True)

    def __add__(self, other):
        return self._binary_operator("add", other)

    def __radd__(self, other):
        return self._binary_operator("add", other, reflect=True)

    def sub(self, other, fill_value=None):
        return self._binary_operator("sub", other, fill_value=fill_value)

    def rsub(self, other, fill_value=None):
        return self._binary_operator("sub", other, fill_value=fill_value, reflect=True)

    def __sub__(self, other):
        return self._binary_operator("sub", other)

    def __rsub__(self, other):
        return self._binary_operator("sub", other, reflect=True)

    def mul(self, other, fill_value=None):
        return self._binary_operator("mul", other, fill_value=fill_value)

    def rmul(self, other, fill_value=None):
        return self._binary_operator("mul", other, fill_value=fill_value, reflect=True)

    def __mul__(self, other):
        return self._binary_operator("mul", other)

    def __rmul__(self, other):
        return self._binary_operator("mul", other, reflect=True)

    def floordiv(self, other, fill_value=None):
        return self._binary_operator("floordiv", other, fill_value=fill_value,)

    def rfloordiv(self, other, fill_value=None):
        return self._binary_operator("floordiv", other, fill_value=fill_value, reflect=True)

    def __floordiv__(self, other):
        return self._binary_operator("floordiv", other)

    def __rfloordiv__(self, other):
        return self._binary_operator("floordiv", other, reflect=True)

    def truediv(self, other, fill_value=None):
        return self._binary_operator("truediv", other, fill_value=fill_value,)

    def rtruediv(self, other, fill_value=None):
        return self._binary_operator("truediv", other, fill_value=fill_value, reflect=True)

    def __truediv__(self, other):
        return self._binary_operator("truediv", other)

    def __rtruediv__(self, other):
        return self._binary_operator("truediv", other, reflect=True)

    # No vectorized shift support, or?
    # def __lshift__(self, other):
    #     return self._binary_operator("lshift", other)

    # def __rlshift__(self, other):
    #     return self._binary_operator("rlshift", other,reflect=True)

    def mod(self, other, fill_value=None):
        return self._binary_operator("mod", other, fill_value=fill_value)

    def rmod(self, other, fill_value=None):
        return self._binary_operator("mod", other, fill_value=fill_value, reflect=True)

    def __mod__(self, other):
        return self._binary_operator("mod", other)

    def __rmod__(self, other):
        return self._binary_operator("mod", other, reflect=True)

    def pow(self, other, fill_value=None):
        return self._binary_operator("pow", other, fill_value=fill_value,)

    def rpow(self, other, fill_value=None):
        return self._binary_operator("pow", other, fill_value=fill_value, reflect=True)

    def __pow__(self, other):
        return self._binary_operator("pow", other)

    def __rpow__(self, other):
        return self._binary_operator("pow", other, reflect=True)

    # comparison

    def eq(self, other, fill_value=None):
        return self._binary_operator("eq", other, fill_value=fill_value)

    def __eq__(self, other):
        return self._binary_operator("eq", other)

    def ne(self, other, fill_value=None):
        return self._binary_operator("ne", other, fill_value=fill_value)

    def __ne__(self, other):
        return self._binary_operator("ne", other)

    def lt(self, other, fill_value=None):
        return self._binary_operator("lt", other, fill_value=fill_value)

    def __lt__(self, other):
        return self._binary_operator("lt", other)

    def gt(self, other, fill_value=None):
        return self._binary_operator("gt", other, fill_value=fill_value)

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

    def _lift(self, func, /, kwargs):
        res = DataFrame()
        if self._null_count == 0:
            for n, c in self._field_data.items():
                res[n] = func(c)(**kwargs)
            return res
        raise NotImplementedError('Dataframe row is not allowed to have nulls')

    def _lift_pairs(self, func, other, /, kwargs):
        res = DataFrame()
        if self._null_count == 0:
            for n, c in self._field_data.items():
                res[n] = func(c)(** {'other': other[n], **kwargs})
            return res
        raise NotImplementedError('Dataframe row is not allowed to have nulls')

    def _unary_operator(self, operator):
        res = DataFrame()
        if self._null_count == 0:
            for n, c in self._field_data.items():
                res[n] = c._unary_operator(operator, c.dtype)
            return res
        raise NotImplementedError('Dataframe row is not allowed to have nulls')

    def _binary_operator(self, operator, other, fill_value=None, reflect=False):

        if isinstance(other, ScalarTypeValues):
            return self._lift(lambda c: c._binary_operator, {'operator': operator,  'other': other, 'fill_value': fill_value, 'reflect': reflect})
        elif isinstance(other, DataFrame):  # order important
            return self._lift_pairs(lambda c: c._binary_operator, other, {'operator': operator,  'fill_value': fill_value, 'reflect': reflect})
        elif isinstance(other, AbstractColumn):  # order important
            # replicate column to match Dataframe:
            other_replicated = DataFrame()
            for n in self._field_data.keys():
                other_replicated[n] = other
            return self._lift_pairs(lambda c: c._binary_operator, other_replicated, {'operator': operator,  'fill_value': fill_value, 'reflect': reflect})

        else:
            raise TypeError(
                f"cannot apply '{operator}' on {type(self).__name__} and {type(other).__name__}")

    # isin ---------------------------------------------------------------

    def isin(self, values: Union[list, "DataFrame", dict]):
        """Check whether values are contained in data."""
        res = DataFrame()
        if isinstance(values, Iterable):
            return self._lift(lambda c: c.isin, {'values': values})
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
                f'isin undefined for values of type {type(self).__name__}.')

    # data cleaning -----------------------------------------------------------
    def fillna(self, fill_value: Union[ScalarTypes, Dict, AbstractColumn, Literal[None]]):
        if fill_value is None:
            return self

        if isinstance(fill_value, ScalarTypeValues):
            return self._lift(lambda c: c.fillna, {'fill_value': fill_value})
        elif isinstance(fill_value, dict):
            res = DataFrame()
            for n in fill_value.keys():
                _ = self._field_data[n]  # throw key error for undefined keys
            for n, c in self._field_data.items():
                res[n] = c.fillna(fill_value=fill_value[n]) if n in dict else c
            return res
        elif isinstance(fill_value, DataFrame):
            res = DataFrame()
            if self._shapeix != fill_value._shapeix:
                TypeError(
                    f"fillna between differently 'shaped' and 'indexed' dataframes is not supported"
                )

            for n, c, d in zip(self._field_data.items(), fill_value.values()):
                res[n] = c.fillna(fill_value=d)
            return res
        else:
            raise TypeError(
                f"fillna with {type(fill_value)} is not supported"
            )

    def dropna(self, how: Literal['any', 'all'] = 'any'):
        """Return a Frame with rows removed where the has any or all nulls."""
        # TODO only flat columns supported...
        res = DataFrame(self._dtype.constructor(nullable=False))
        if how == 'any':
            for i in self:
                if not DataFrame._has_any_null(i):
                    res._append(i)
        elif how == 'all':
            for i in self:
                if not DataFrame._has_all_null(i):
                    res._append(i)
        return res

    def drop_duplicates(self,
                        subset: Union[str, List[str], Literal[None]] = None,
                        keep: Literal['first', 'last', False] = "first"):
        """ Remove duplicate values from data but keep the first, last, none (keep=False)"""
        # Todo Add functionality
        assert keep == 'first'
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

    def min(self, numeric_only=None):
        """Return the minimum of the nonnull values of the Column."""
        return self._summarize(lambda c: c.min, {'numeric_only': numeric_only})

    def max(self, numeric_only=None):
        """Return the maximum of the nonnull values of the column."""
        # skipna == True
        return self._summarize(lambda c: c.max, {'numeric_only': numeric_only})

    def all(self, boolean_only=None):
        """Return whether all nonull elements are True in Column"""
        return self._summarize(lambda c: c.all, {'boolean_only': boolean_only})

    def any(self, skipna=True, boolean_only=None):
        """Return whether any nonull element is True in Column"""
        return self._summarize(lambda c: c.any,  {'boolean_only': boolean_only})

    def sum(self):
        """Return sum of all nonull elements in Column"""
        return self._summarize(lambda c: c.sum, {})

    def prod(self):
        """Return produce of the values in the data"""
        return self._summarize(lambda c: c.prod, {})

    def cummin(self, skipna=True):
        """Return cumulative minimum of the data."""
        return self._lift(lambda c: c.cummin, {'skipna': skipna})

    def cummax(self, skipna=True):
        """Return cumulative maximum of the data."""
        return self._lift(lambda c: c.cummax, {'skipna': skipna})

    def cumsum(self, skipna=True):
        """Return cumulative sum of the data."""
        return self._lift(lambda c: c.cumsum, {'skipna': skipna})

    def cumprod(self, skipna=True):
        """Return cumulative product of the data."""
        return self._lift(lambda c: c.cumprod, {'skipna': skipna})

    def mean(self):
        """Return the mean of the values in the series."""
        return self._summarize(lambda c: c.mean, {})

    def median(self):
        """Return the median of the values in the data."""
        return self._summarize(lambda c: c.median, {})

    def mode(self):
        """Return the mode(s) of the data."""
        return self._summarize(lambda c: c.mode, {})

    def std(self):
        """Return the stddev(s) of the data."""
        return self._summarize(lambda c: c.std, {})

    def nunique(self, dropna=True):
        """Returns the number of unique values per column"""
        res = DataFrame(
            Struct([Field('column', string), Field('nunique', int64)]))
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
            includes = [n for n, c in self._field_data.items()
                        if is_numerical(c.dtype)]
        elif isinstance(include_columns, list):
            includes = [n for n, c in self._field_data.items()
                        if c.dtype in include_columns]
        else:
            raise TypeError(
                f"describe with include_columns of type {type(include_columns).__name__} is not supported"
            )

        excludes = []
        if exclude_columns is None:
            excludes = []
        elif isinstance(exclude_columns, list):
            excludes = [n for n, c in self._field_data.items()
                        if c.dtype in exclude_columns]
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
        res['metric'] = ['count', 'mean', 'std', 'min'] + \
            [f'{p}%' for p in percentiles] + ['max']
        for s in selected:
            c = self._field_data[s]
            res[s] = [c.count(), c.mean(), c.std(), c.min()] + \
                c._percentiles(percentiles) + [c.max()]
        return res

    # Dataframe specific ops --------------------------------------------------    #

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

    def keep(self, columns: List[str]):
        """
        Returns DataFrame with the kept columns only.
        """
        print("KEEP", self, columns)
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

    def rename(self, column_mapper: Dict[str, str]):
        if isinstance(column_mapper, dict):
            for n in column_mapper:
                _ = self._field_data[n]  # creates key error
            res = DataFrame()
            for n, c in self._field_data.items():
                if n in column_mapper:
                    res[column_mapper[n]] = c
            return res
        else:
            raise TypeError(
                f"rename with column_mapper parameter of type {type(column_mapper).__name__} is not supported"
            )

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
        import pandas as pd
        map = {}
        for n, c in self._field_data.items():
            map[n] = c.to_pandas()
        return pd.DataFrame(map)

    def to_arrow(self):
        """Convert self to arrow table"""
        # TODO Add type translation
        import pyarrow as pa
        map = {}
        for n, c in self._field_data.items():
            map[n] = c.to_arrow()
        return pa.table(map)

    # fluent with symbolic expressions ----------------------------------------

    def _where(self, *conditions):
        """
        Analogous to SQL's where.

        Filter a dataframe to only include
        rows satisfying a given set of conditions.
        """
        from .symexp import Symbol, eval_symbolic

        if not conditions:
            return self

        evalled_conditions = [eval_symbolic(condition, {'me': self})
                              for condition in conditions]
        anded_evalled_conditions = functools.reduce(
            lambda x, y: x & y, evalled_conditions)
        return self[anded_evalled_conditions]

    def _select(self, *args, **kwargs):
        """
        Analogous to SQL's ``SELECT`.

        Transform a dataframe by selecting old columns and new (computed)
        columns.
        """
        from .symexp import Symbol, eval_symbolic

        input_columns = set(self.columns)
        # print("SELECT", self, input_columns, '|', args, kwargs)

        has_star = False
        include_columns = []
        exclude_columns = []
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError('args must be column names')
            if arg == '*':
                if has_star:
                    raise ValueError('select received repeated stars')
                has_star = True
            elif arg in input_columns:
                if arg in include_columns:
                    raise ValueError(
                        f'select received a repeated column-include ({arg})')
                include_columns.append(arg)
            elif arg[0] == '-' and arg[1:] in input_columns:
                if arg in exclude_columns:
                    raise ValueError(
                        f'select received a repeated column-exclude ({arg[1:]})')
                exclude_columns.append(arg[1:])
            else:
                raise ValueError(
                    f'argument ({arg}) does not denote an existing column')
        if exclude_columns and not has_star:
            raise ValueError(
                'select received column-exclude without a star')
        if has_star and include_columns:
            raise ValueError(
                'select received both a star and column-includes')
        if set(include_columns) & set(exclude_columns):
            raise ValueError(
                'select received overlapping column-includes and ' +
                'column-excludes')

        include_columns_inc_star = self.columns if has_star else include_columns

        output_columns = [col for col in include_columns_inc_star
                          if col not in exclude_columns]

        res = DataFrame()
        for n, c in self._field_data.items():
            if n in output_columns:
                res[n] = c
        for n, c in kwargs.items():
            res[n] = eval_symbolic(c, {'me': self})
        return res

    def pipe(self, func, *args, **kwargs):
        """
        Apply func(self, \*args, \*\*kwargs).
        """
        return func(self, *args, **kwargs)

#     def _groupby(self, by: List[str],
#             sort=False,
#             dropna=True,
#         ):
#         grouping_columns = by
#         colected_columns =[]
#         for tup in zip([self_field_dat])

#         for

# @dataclass
# class GroupedBy:
#     groups: Dict
#     def agg(self, *args, **kwargs):

#         res = Datafarme()
#         for n, c in self._field_data.items():
#             if n in output_columns:
#                 res[n] = c
#         for n, c in kwargs.item():
#             res[n] = eval_symbolic(c, {'me': self})
#         return res


# class dataFRame
# ------------------------------------------------------------------------------


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
# Relational operators, still TBD


# @annotate("JOIN", color="blue", domain="cudf_python")
#     def merge(
#         self,
#         right,
#         on=None,
#         left_on=None,
#         right_on=None,
#         left_index=False,
#         right_index=False,
#         how="inner",
#         sort=False,
#         lsuffix=None,
#         rsuffix=None,
#         method="hash",
#         indicator=False,
#         suffixes=("_x", "_y"),
#     ):
#         """Merge GPU DataFrame objects by performing a database-style join
#         operation by columns or indexes."""


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
# def info(
#         self,
#         verbose=None,
#         buf=None,
#         max_cols=None,
#         memory_usage=None,
#         null_counts=None,
#     ):
#         """
#         Print a concise summary of a DataFrame.
#         This method prints information about a DataFrame including
#         the index dtype and column dtypes, non-null values and memory usage.

#         """"
