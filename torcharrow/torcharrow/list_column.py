import array as ar
import builtins
import copy
import functools
from dataclasses import dataclass
from typing import List, Optional

from .column import (AbstractColumn, Column, _column_constructor,
                     _set_column_constructor)
from .dtypes import (NL, Boolean, DType, Int64, List_, String, Uint32, is_list,
                     is_string)
from .tabulate import tabulate

# -----------------------------------------------------------------------------
# ListColumn


class ListColumn(AbstractColumn):

    def __init__(self,  dtype: List_, kwargs=None):
        assert is_list(dtype)
        super().__init__(dtype)
        self._data = _column_constructor(dtype.item_dtype)
        self._offsets = ar.array('I', [0])  # Uint32
        self.list = ListMethods(self)

    def _invariant(self):
        assert len(self._data) == len(self._validity)
        assert len(self._data) == len(self._offsets)-1
        assert self._offsets[-1] == len(self._data)
        assert all(self._offsets[i] <= self._offsets[i+1]
                   for i in range(0, len(self._offsets)+1))
        assert 0 <= self._offset and self._offset <= len(self._data)
        assert 0 <= self._length and self._offset + \
            self._length <= len(self._data)
        rng = range(self._offset, self._offset+self._length)
        assert self.null_length == sum(
            self._validity[self._offset+i] for i in rng)

    # implementing abstract methods ----------------------------------------------

    def _raw_lengths(self):
        return self._data._raw_lengths()

    @property
    def is_appendable(self):
        """Can this column/frame be extended without side effecting """
        return self._data.is_appendable and len(self._data) == self._offsets[self._offset + self._length]

    def memory_usage(self, deep=False):
        """Return the memory usage of the column/frame (if deep then include buffer sizes)."""
        osize = self._offsets.itemsize
        vsize = self._validity.itemsize
        dusage = self._data.memory_usage(deep)
        if not deep:
            nchars = (self._offsets[self._offset +
                      self.length]-self._offsets[self._offset])
            return self._length * vsize + self._length * osize + dusage
        else:
            return len(self._validity)*vsize + len(self.self._offsets)*osize + dusage

    def _copy(self, deep, offset, length):
        if deep:
            res = ListColumn(self.dtype)
            res._length = length
            res._data = self._data[self._offsets[self._offset]: self._offsets[offset+length]]
            res._validity = self._validity[offset: offset+length]
            res._offsets = self._offsets[offset: offset+length+1]
            res._null_count = sum(res._validity)
            return res
        else:
            return copy.copy(self)

    def _append(self, values):
        if values is None:
            if self._dtype.nullable:
                self._null_count += 1
                self._validity.append(False)
                self._offsets.append(self._offsets[-1])
            else:
                raise TypeError("a list is required (got type NoneType)")
        else:
            self._validity.append(True)
            self._data.extend(values)
            self._offsets.append(self._offsets[-1]+len(values))
        self._length += 1

    def get(self, i, fill_value):
        """Get ith row from column/frame"""
        j = self._offset+i
        if not self._validity[j]:
            return fill_value
        else:
            return list(self._data[self._offsets[j]:self._offsets[j+1]])

    def __iter__(self):
        for i in range(self._length):
            j = self._offset+i
            if self._validity[j]:
                yield list(self._data[self._offsets[j]:self._offsets[j+1]])
            else:
                yield None

    # printing ----------------------------------------------------------------
    def __str__(self):
        return f"Column([{', '.join('None' if i is None else str(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate([['None' if i is None else str(i)]
                       for i in self], tablefmt='plain', showindex=True)
        typ = f"dtype: {self._dtype}, length: {self._length}, null_count: {self._null_count}"
        return tab+NL+typ

    def show_details(self):
        return _Repr(self)


@dataclass
class _Repr:
    parent: ListColumn

    def __repr__(self):
        raise NotImplementedError()
        # #TODO
        # me = self.parent
        # tab = tabulate([[l if l is not None else 'None', v] for (l,o, v) in zip(me._data, me._offsets, me._validity)],['data', 'offsets', 'validity'])
        # typ = f"dtype: {me._dtype}, count: {me._length}, null_count: {me._null_count}, offset: {me._offset}"
        # return tab+NL+typ


# ------------------------------------------------------------------------------
# registering the factory
_set_column_constructor(is_list, ListColumn)


def __map(fun, xs):
    return builtins.map(fun, xs)

# -----------------------------------------------------------------------------
# ListMethods


@dataclass(frozen=True)
class ListMethods:
    """Vectorized list functions for ListColumn"""
    _parent: ListColumn

    def _map_map(self, fun, dtype: Optional[DType] = None):
        func = fun
        me = self._parent
        assert dtype is not None
        res = _column_constructor(dtype)
        for i in range(me._length):
            j = me._offset+i
            if me._validity[j]:
                res.append(fun(me[j]))
            else:
                res.append(None)
        return res

    def join(self, sep):
        """Join lists contained as elements with passed delimiter."""
        me = self._parent
        assert is_string(me.dtype.item_dtype)
        def fun(i): return sep.join(i)
        return self._map_map(fun, String(me.dtype.item_dtype.nullable or me.dtype.nullable))

    def get(self, i):
        me = self._parent
        def fun(xs): return xs[i]
        return self._map_map(fun, me.dtype.item_dtype)

    def count(self, elem, flags=0):
        me = self._parent
        def fun(i): return i.count(elem)
        return self._map_map(fun, Int64(me.dtype.nullable))

    def map(self, fun, dtype: Optional[DType] = None):
        me = self._parent

        def func(xs): return list(builtins.map(fun, xs))
        if dtype is None:
            dtype = me.dtype
        return self._map_map(func, dtype)

    def filter(self, pred):
        me = self._parent
        def func(xs): return list(builtins.filter(pred, xs))
        return self._map_map(func, me.dtype)

    def reduce(self, fun, initializer=None, dtype: Optional[DType] = None):
        me = self._parent
        def func(xs): return functools.reduce(fun, xs, initializer)
        dtype = me.dtype.item_dtype if dtype is None else dtype
        return self._map_map(func, dtype)

    def flatmap(self, fun, dtype: Optional[DType] = None):
        # dtype must be given, if result is different from argument column
        me = self._parent

        def func(xs):
            ys = []
            for x in xs:
                ys.extend(fun(x))
            return ys
        return self._map_map(func, me.dtype)


# ops on list  --------------------------------------------------------------
#  'count',
#  'extend',
#  'index',
#  'insert',
#  'pop',
#  'remove',
#  'reverse',
