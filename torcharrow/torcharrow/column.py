import array as ar
import functools
import math
import operator
import statistics
from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict
from typing import (Callable, Dict, Iterable, List, Literal,  Optional,
                    Sequence, Sized, Tuple, Union, Iterable)

from .dtypes import (DType, Field, Float64, Int64, ScalarTypes,
                     ScalarTypeValues, Struct, boolean, derive_dtype,
                     derive_operator, float64, infer_dtype_from_prefix,
                     is_boolean, is_numerical, is_primitive, is_struct,
                     is_tuple, string)

# ------------------------------------------------------------------------------
# Column - the factory method for specialized columns.


def Column(data: Union[Iterable, DType, Literal[None]] = None, dtype: Optional[DType] = None):
    """
    Column factory method; returned column type has elements of homogenous dtype.    
    """

    # handling cyclic references...
    from .list_column import ListColumn
    from .map_column import MapColumn
    from .numerical_column import NumericalColumn
    from .string_column import StringColumn

    if data is None and dtype is None:
        raise TypeError('Column requires data and/or dtype parameter')

    if data is not None and isinstance(data, DType):
        if dtype is not None and isinstance(dtype, DType):
            raise TypeError('Column can only have one dtype parameter')
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
                f'dtype parameter of DType expected (got {type(dtype).__name__})')

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
                raise TypeError('Column cannot infer type from data')
            if is_tuple(dtype):
                raise TypeError(
                    'Column cannot be used to created structs, use Dataframe constructor instead')
            col = _column_constructor(dtype)
            # add prefix and ...
            col.extend(prefix)
            # ... continue enumerate the data
            for _, v in enumerate(data):
                col.append(v)
            return col
        else:
            raise TypeError(
                f'data parameter of Sequence type expected (got {type(dtype).__name__})')
    else:
        raise AssertionError('unexpected case')


@staticmethod
def arange(
        start: Union[int, float],
        stop: Union[int, float, None] = None,
        step: Union[int, float] = 1,
        dtype: Optional[DType] = None) -> Column:
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
    raise KeyError(f'no matching test found for {dtype}')

# ------------------------------------------------------------------------------
# AbstractColumn


class AbstractColumn(ABC, Sized, Iterable):
    """AbstractColumn are one dimenensionalcolumns or two dimensional dataframes"""

    def __init__(self, dtype: DType):
        self._dtype = dtype
        self._offset = 0
        self._length = 0
        self._null_count = 0
        self._validity = ar.array('b')

     # simple meta data getters -----------------------------------------------

    @property
    def dtype(self):
        """dtype of the colum/frame"""
        return self._dtype

    @property
    def isnullable(self):
        """A boolean indicating whether column/frame can have nulls"""
        return self.dtype.nullable

    @abstractproperty
    def is_appendable(self):
        """Can this column/frame be extended without side effecting """
        pass

    def count(self):
        """Return number of non-NA/null observations pgf the column/frame"""
        return self._length - self._null_count

    def null_count(self):
        """Number of null values"""
        return self._null_count

    def __len__(self):
        """Return number of rows including null values"""
        return self._length

    @property
    def ndim(self):
        """Column ndim is always 1, Frame ndim is always 2"""
        return 1

    @property
    def size(self):
        """Number of rows * number of columns."""
        return len(self)

    @abstractmethod
    def _raw_lengths(self) -> List[int]:
        "Lengths of underlying buffers"
        pass

    def _valid(self, i):
        return self._validity[self._offset+i]

    # builders and iterators---------------------------------------------------
    def append(self, value):
        """Append value to the end of the column/frame"""
        if not self.is_appendable:
            raise AttributeError('column is not appendable')
        else:
            return self._append(value)

    @abstractmethod
    def _append(self, value):
        pass

    def extend(self, iterable: Iterable):
        """Append items from iterable to the end of the column/frame"""
        if not self.is_appendable:
            raise AttributeError('column is not appendable')
        for i in iterable:
            self._append(i)

    def concat(self, others: List["AbstractColumn"]):
        """Concatenate columns/frames."""
        # instead of pandas.concat
        for o in others:
            self.extend(o)

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

    def __getitem__(self, arg):
        """
        If *arg* is a ``str`` type, return the str'th column (only defined for Frames).
        If *arg* is a ``int`` type, return a Frame with the int'th row.
        If *arg* is a ``slice`` of column names, return a new Frame with all columns
        sliced to the specified range.
        If *arg* is a ``slice`` of ints, return a new DataFrame with all rows
        sliced to the specified range.
        If *arg* is an ``list`` containing column names, return a new
        DataFrame with the corresponding columns.
        If *arg* is an ``list`` containing row numbers, return a new
        DataFrame with the corresponding rows.
        If *arg* is a ``BooleanColumn``, return the rows marked True
        """
        # print('slice', arg, str(type(arg)))
        if isinstance(arg, int):
            return self._get_row(arg)
        elif isinstance(arg, str):
            return self._get_column(arg)
        elif isinstance(arg, slice):
            if isinstance(arg.start, str) or isinstance(arg.stop, str) or isinstance(arg.step, str):
                return self._slice_columns(arg)
            elif isinstance(arg.start, int) or isinstance(arg.stop, int) or isinstance(arg.step, int):
                return self._slice_rows(arg)
        if isinstance(arg, (tuple, list)):
            if len(arg) == 0:
                return self._empty()
            elif isinstance(arg[0], str) and all(isinstance(a, str) for a in arg):
                return self._pick_columns(arg)
            elif isinstance(arg[0], int) and all(isinstance(a, int) for a in arg):
                return self._pick_rows(arg)
            else:
                raise TypeError('index should be list of int or list of str')
            return self._get_columns_by_label(arg, downcast=True)

        elif isinstance(arg, AbstractColumn) and is_boolean(arg.dtype):
            return self.filter(arg)
        else:
            raise TypeError(
                f"__getitem__ on type {type(arg)} is not supported"
            )

    def slice(self, start=None, stop=None, step=1):
        return self._slice_rows(slice(start, stop, step))

    def head(self, n=5):
        """Return the first `n` rows."""
        return self._slice_rows(slice(0, n, None))

    def tail(self, n=5):
        """Return the last `n` rows."""
        return self._slice_rows(slice(-n, len(self), None))

    def reverse(self):
        res = _column_constructor(self.dtype)
        for i in range(len(self)):
            res._append(self[(len(self)-1)-i])
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
        # TODO compare Pandas/TorchArrow/Python slices, last index is included or not?
        # here like Python!
        # print('slice_rows', arg, str(type(arg)))

        step = 1
        if arg.step is not None:
            step = arg.step

        start = 0
        if arg.start is not None:
            if arg.start >= 0:
                start = arg.start
            elif arg.start < 0:
                start = arg.start + len(self)

        stop = len(self)
        if arg.stop is not None:
            if arg.stop >= 0:
                stop = arg.stop
            elif arg.stop < 0 and not (step < 0 and arg.stop == -1):
                stop = arg.stop + len(self)

        # print('_slice_rows', start, stop, step)
        # empty slice
        if (step > 0 and start >= stop) or (step < 0 and start <= stop):
            return _column_constructor(self.dtype)

        # create a view
        if start <= stop and step == 1:
            res = self.copy(deep=False)
            res._offset = res._offset+start
            res._length = stop-start
            # update null_count
            res._nullcount = sum(self._valid(i) for i in range(len(self)))
            return res

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

    def to_pandas(self, nullable=False):
        """Convert to a Pandas data."""
        raise NotImplementedError()

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
    def map(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None] = None,
        dtype: Optional[DType] = None,
    ):
        """
        Map rows according to input correspondence.
        dtype required if result type != item type.
        """
        if isinstance(arg, dict):
            return self._map(lambda x: arg.get(x, None), na_action, dtype)
        else:
            return self._map(arg, na_action, dtype)

    def flatmap(
        self,
        arg: Union[Dict, Callable],
        na_action: Literal["ignore", None] = None,
        dtype: Optional[DType] = None,
    ):
        """
        Map rows to list of rows according to input correspondance
        dtype required if result type != item type.
        """
        def func(x): return arg.get(
            x, None) if isinstance(arg, dict) else arg(x)
        dtype = dtype if dtype is not None else self._dtype
        res = _column_constructor(dtype)
        for i in range(self._length):
            if self._valid(i) or na_action == "ignore":
                res.extend(func(self[i]))
            else:
                res._append(None)
        return res

    def filter(self, predicate: Union[Callable, Iterable]):
        """
        Select rows where predicate is True.
        Different from Pandas. Use keep for Pandas filter.
        """
        if not isinstance(predicate, (Callable, Iterable)):
            raise TypeError(
                "predicate must be a unary boolean predicate or iterable of booleans"
            )
        res = _column_constructor(self._dtype)
        if isinstance(predicate, Callable):
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

    def reduce(self, fun, initializer=None):
        """
        Apply binary function cumulatively to the rows[0:],
        so as to reduce the column/dataframe to a single value
        """
        if self._length == 0:
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
        for i in range(start, self._length):
            value = fun(value, self[i])
        return value

    def _map(
        self,
        func: Callable,
        na_action: Literal["ignore", None] = None,
        dtype: Optional[DType] = None,
    ):
        dtype = dtype if dtype is not None else self._dtype

        res = _column_constructor(dtype)
        for i in range(self._length):
            if self._valid(i) or na_action == "ignore":
                res._append(func(self[i]))
            else:
                res._append(None)
        return res

    # ifthenelse -----------------------------------------------------------------

    def where(self, condition, other):
        """Equivalent to ternary expression: if condition then self else other"""
        if not isinstance(condition, Iterable):
            raise TypeError("condition must be an iterable of booleans")

        res = _column_constructor(self._dtype)
        # check for length?
        if isinstance(other, ScalarTypeValues):
            for s, m in zip(self, condition):
                if m:
                    res._append(s)
                else:
                    res._append(other)
            return res
        elif isinstance(other, Iterable):
            for s, m, o in zip(self, condition, other):
                if m:
                    res._append(s)
                else:
                    res._append(o)
            return res
        else:
            raise TypeError(f"where on type {type(other)} is not supported")

    def mask(self, condition, other):
        """Equivalent to ternary expression: if ~condition then self else other."""
        return self.where(~condition, other)

    # sorting -----------------------------------------------------------------

    def sort_values(
        self,
        by: Union[str, List[str], Literal[None]] = None,
        ascending=True,
        na_position: Literal['last', 'first'] = "last",

    ):
        """Sort a column/a dataframe in ascending or descending order"""
        # key:Callable, optional missing
        if by is not None:
            raise TypeError(
                "sorting a non-structured column can't have 'by' parameter")
        res = _column_constructor(self.dtype)
        if na_position == 'first':
            res.extend([None] * self._null_count)
        res.extend(
            sorted((i for i in self if i is not None), reverse=not ascending))
        if na_position == 'last':
            res.extend([None] * self._null_count)
        return res

    def nlargest(self, n=5,  columns: Union[str, List[str], Literal[None]] = None, keep: Literal['last', 'first'] = "first"):
        """Returns a new data of the *n* largest element."""
        # keep="all" not supported
        # pass columns for comparison , nlargest(by=['country', ..]..)
        # first/last matters only for dataframes with columns
        # Todo add keep arg
        if columns is not None:
            raise TypeError(
                "computing n-largest on non-structured column can't have 'columns' parameter")
        return self.sort_values(ascending=False).head(n)

    def nsmallest(self, n=5, columns: Union[str, List[str], Literal[None]] = None, keep="first"):
        """Returns a new data of the *n* smallest element. """
        # keep="all" not supported
        # first/last matters only for dataframes with columns
        # Todo add keep arg
        if columns is not None:
            raise TypeError(
                "computing n-smallest on non-structured column can't have 'columns' parameter")

        return self.sort_values(ascending=True).head(n)

    def nunique(self, dropna=True):
        """Returns the number of unique values of the Column"""
        if not dropna:
            return len(set(self))
        else:
            return len(set(i for i in self if i is not None))

    # operators ---------------------------------------------------------------

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
        return self._binary_operator("floordiv",  other, fill_value=fill_value)

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
        return self._unary_operator(operator.__not__, self.dtype)

    # unary arithmetic
    def __neg__(self):
        return self._unary_operator(operator.neg, self.dtype)

    def __pos__(self):
        return self._unary_operator(operator.pos, self.dtype)

    # unary math related

    def abs(self):
        """Absolute value of each element of the series."""
        return self._unary_operator(abs, self.dtype)

    def ceil(self):
        """Rounds each value upward to the smallest integral"""
        return self._unary_operator(math.ceil, Int64(self.dtype.nullable))

    def floor(self):
        """Rounds each value downward to the largest integral value"""
        return self._unary_operator(math.floor, Int64(self.dtype.nullable))

    def round(self, decimals: Optional[int] = None):
        """Round each value in a data to the given number of decimals."""
        if decimals is None:
            return self._unary_operator(round, Int64(self.dtype.nullable))
        else:
            def round_(x): return round(x, decimals)
            return self._unary_operator(round_, Float64(self.dtype.nullable))

    def hash_values(self):
        """Compute the hash of values in this column."""
        return self._unary_operator(hash, Int64(self.dtype.nullable))

    def _broadcast(self, operator, const,  fill_value, dtype, reflect):
        assert is_primitive(self._dtype) and is_primitive(dtype)
        # print('Column._broadcast', self, operator, const, fill_value, dtype, reflect)
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

    def _pointwise(self, operator, other,  fill_value, dtype, reflect):
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

    def _unary_operator(self, operator, dtype):
        res = _column_constructor(dtype)
        for i in range(self._length):
            if self._valid(i):
                res._append(operator(self[i]))
            else:
                res._append(None)
        return res

    def _binary_operator(self, operator, other, fill_value=None, reflect=False):
        if isinstance(other, (int, float, list, set, type(None))):
            return self._broadcast(derive_operator(operator), other,  fill_value,  derive_dtype(self._dtype, operator), reflect)
        elif is_struct(other.dtype):
            raise TypeError(
                f"cannot apply '{operator}' on {type(self).__name__} and {type(other).__name__}")
        else:
            return self._pointwise(derive_operator(operator), other,   fill_value, derive_dtype(self._dtype, operator), reflect)

    # isin --------------------------------------------------------------------
    def isin(self, values: Union[list, dict, "AbstractColumn"]):
        """Check whether list values are contained in data, or column/dataframe (row/column specific)."""
        if isinstance(values, list):
            res = _column_constructor(boolean)
            for i in self:
                res._append(i in values)
            return res
        else:
            raise ValueError(
                f'isin undefined for values of type {type(self).__name__}.')

    # data cleaning -----------------------------------------------------------

    def fillna(self, fill_value: Union[ScalarTypes, Dict, "AbstractColumn", Literal[None]]):
        """Fill NA/NaN values with scalar (like 0) or column/dataframe (row/column specific)"""
        if fill_value is None:
            return self
        res = _column_constructor(self._dtype.constructor(nullable=False))
        if isinstance(fill_value, ScalarTypeValues):
            for value in self:
                if value is not None:
                    res._append(value)
                else:
                    res._append(fill_value)
            return res
        elif isinstance(fill_value, Column):  # TODO flat Column   --> needs a test
            for value, fill in zip(self, fill_value):
                if value is not None:
                    res._append(value)
                else:
                    res._append(fill)
            return res
        else:  # Dict and Dataframe only self is dataframe
            raise TypeError(
                f"fillna with {type(fill_value)} is not supported"
            )

    def dropna(self, how: Literal['any', 'all'] = any):
        """Return a column/frame with rows removed where a row has any or all nulls."""
        # TODO only flat columns supported...
        # notet hat any and all play nor role for flat columns,,,
        res = _column_constructor(self._dtype.constructor(nullable=False))
        for i in range(self._offset, self._offset+self._length):
            if self._validity[i]:
                res._append(self[i])
        return res

    def drop_duplicates(self, subset: Union[str, List[str], Literal[None]] = None, keep: Literal['first', 'last', False] = "first"):
        """ Remove duplicate values from row/frame but keep the first, last, none"""
        # Todo Add functionality
        assert keep == 'first'
        if subset is not None:
            raise TypeError(
                f"subset parameter for flat columns not supported"
            )
        res = _column_constructor(self._dtype)
        res.extend(list(OrderedDict.fromkeys(self)))
        return res

    # universal  ---------------------------------------------------------------

    def min(self, numeric_only=None):
        """Return the minimum of the nonnull values of the Column."""
        # skipna == True
        # default implmentation:
        if numeric_only is None or (numeric_only and is_numerical(self.dtype)):
            return min(self._iter(skipna=True))
        else:
            raise ValueError(f'min undefined for {type(self).__name__}.')

    def max(self, numeric_only=None):
        """Return the maximum of the nonnull values of the column."""
        # skipna == True
        if numeric_only is None or (numeric_only and is_numerical(self.dtype)):
            return max(self._iter(skipna=True))
        else:
            raise ValueError(f'max undefined for {type(self).__name__}.')

    def all(self, boolean_only=None):
        """Return whether all nonull elements are True in Column"""
        # skipna == True
        if boolean_only is None or (boolean_only and is_boolean(self.dtype)):
            return all(self._iter(skipna=True))
        else:
            raise ValueError(f'all undefined for {type(self).__name__}.')

    def any(self, skipna=True, boolean_only=None):
        """Return whether any nonull element is True in Column"""
        # skipna == True
        if boolean_only is None or (boolean_only and is_boolean(self.dtype)):
            return any(self._iter(skipna=True))
        else:
            raise ValueError(f'all undefined for {type(self).__name__}.')

    def sum(self):
        """Return sum of all nonull elements in Column"""
        # skipna == True
        # only_numerical == True
        if is_numerical(self.dtype):
            return sum(self._iter(skipna=True))
        else:
            raise ValueError(f'max undefined for {type(self).__name__}.')

    def prod(self):
        """Return produce of the values in the data"""
        # skipna == True
        # only_numerical == True
        if is_numerical(self.dtype):
            return functools.reduce(operator.mul, self._iter(skipna=True), 1)
        else:
            raise ValueError(f'prod undefined for {type(self).__name__}.')

    def cummin(self, skipna=True):
        """Return cumulative minimum of the data."""
        # skipna == True
        if is_numerical(self.dtype):
            return self._accumulate_column(min, skipna=skipna, initial=None)
        else:
            raise ValueError(f'cumin undefined for {type(self).__name__}.')

    def cummax(self, skipna=True):
        """Return cumulative maximum of the data."""
        if is_numerical(self.dtype):
            return self._accumulate_column(max, skipna=skipna,  initial=None)
        else:
            raise ValueError(f'cummax undefined for {type(self).__name__}.')

    def cumsum(self, skipna=True):
        """Return cumulative sum of the data."""
        if is_numerical(self.dtype):
            return self._accumulate_column(operator.add,  skipna=skipna, initial=None)
        else:
            raise ValueError(f'cumsum undefined for {type(self).__name__}.')

    def cumprod(self, skipna=True):
        """Return cumulative product of the data."""
        if is_numerical(self.dtype):
            return self._accumulate_column(operator.mul,  skipna=skipna,  initial=None)
        else:
            raise ValueError(f'cumprod undefined for {type(self).__name__}.')

    def mean(self):
        """Return the mean of the values in the series."""
        if is_numerical(self.dtype):
            return statistics.mean(self._iter(skipna=True))
        else:
            raise ValueError(f'mean undefined for {type(self).__name__}.')

    def median(self):
        """Return the median of the values in the data."""
        if is_numerical(self.dtype):
            return statistics.median(self._iter(skipna=True))
        else:
            raise ValueError(f'median undefined for {type(self).__name__}.')

    def mode(self):
        """Return the mode(s) of the data."""
        if is_numerical(self.dtype):
            return statistics.mode(self._iter(skipna=True))
        else:
            raise ValueError(f'mode undefined for {type(self).__name__}.')

    def std(self):
        """Return the stddev(s) of the data."""
        if is_numerical(self.dtype):
            return statistics.stdev(self._iter(skipna=True))
        else:
            raise ValueError(f'std undefined for {type(self).__name__}.')

    def _iter(self, skipna):
        for i in self:
            if not(i is None and skipna):
                yield i

    def _percentiles(self, percentiles):
        if len(self) == 0 or len(percentiles) == 0:
            return []
        out = []
        s = sorted(self)
        for percent in percentiles:
            k = (len(self)-1) * (percent/100)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                out.append(s[int(k)])
                continue
            d0 = s[int(f)] * (c-k)
            d1 = s[int(c)] * (k-f)
            out.append(d0+d1)
        return out

    def _accumulate_column(self, func, *, skipna=True, initial=None):
        it = iter(self)
        res = _column_constructor(self.dtype)
        total = initial
        rest_is_null = False
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                raise ValueError(f'cum[min/max] undefined for empty column.')
        if total is None:
            raise ValueError(
                f'cum[min/max] undefined for columns with row 0 as null.')
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
                f"'include/exclude columns' parameter for '{type(self).__name__}' not supported ")
        if percentiles is None:
            percentiles = [25, 50, 75]
        percentiles = sorted(set(percentiles))
        if len(percentiles) > 0:
            if percentiles[0] < 0 or percentiles[-1] > 100:
                raise ValueError("percentiles must be betwen 0 and 100")

        if is_numerical(self.dtype):
            res = DataFrame(
                Struct([Field('statistic', string), Field('value', float64)]))
            res._append(('count', self.count()))
            res._append(('mean', self.mean()))
            res._append(('std', self.std()))
            res._append(('min', self.min()))
            values = self._percentiles(percentiles)
            for p, v in zip(percentiles, values):
                res._append((f'{p}%', v))
            res._append(('max', self.max()))
            return res
        else:
            raise ValueError(f'median undefined for {type(self).__name__}.')

    # Flat column specfic ops ----------------------------------------------------------
    def is_unique(self):
        """Return boolean if data values are unique."""
        seen = set()
        return not any(i in seen or seen.add(i) for i in self)

    # only on flat column
    def is_monotonic_increasing(self):
        """Return boolean if values in the object are monotonic increasing"""
        return self._compare(operator.lt, initial=True)

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

    def to_pandas(self):
        """Convert selef to pandas dataframe"""
        # TODO Add type translation
        import pandas as pd
        return pd.Series(self)

    def to_arrow(self):
        """Convert selef to pandas dataframe"""
        # TODO Add type translation
        import pyarrow as pa
        return pa.array(self)

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

    # functools ---------------------------------------------------------------

    # def map(self, fun, dtype:Optional[DType]=None):
    #     #dtype must be given, if result is different from argument column
    #     if dtype is None:
    #         dtype = self._dtype
    #     res = _create_column(dtype)
    #     for i in range(self._length):
    #         if self._validity[i]:
    #             res._append(fun(self[i]))
    #         else:
    #             res._append(None)
    #     return res

    # def filter(self, pred):
    #     res = _create_column(self._dtype)
    #     for i in range(self._length):
    #         if self._validity[i]:
    #             if pred(self[i]):
    #                 res._append(self[i])
    #                 continue
    #         else:
    #             res._append(None)
    #     return res

    # def reduce(self, fun, initializer=None):
    #     if self._length==0:
    #         if initializer is not None:
    #             return initializer
    #         else:
    #             raise TypeError("reduce of empty sequence with no initial value")
    #     start = 0
    #     if initializer is None:
    #         value = self[0]
    #         start = 1
    #     else:
    #         value = initializer
    #     for i in range(start,self._length):
    #         value = fun(value, self[i])
    #     return value

    # def flatmap(self, fun, dtype:Optional[DType]=None):
    #     #dtype must be given, if result is different from argument column
    #     if dtype is None:
    #         dtype = self._dtype
    #     res = _create_column(dtype)
    #     for i in range(self._length):
    #         if self._validity[i]:
    #             res.extend(fun(self[i]))
    #         else:
    #             res._append(None)
    #     return res
