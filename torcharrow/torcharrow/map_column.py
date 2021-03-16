import array as ar
import copy
from dataclasses import dataclass
from typing import (Any, Callable, ClassVar, Dict, List, Literal, Optional,
                    TypeVar, Union)

from .column import (AbstractColumn, Column, _column_constructor,
                     _set_column_constructor)
from .dtypes import NL, DType, List_, is_map
from .list_column import ListColumn
from .tabulate import tabulate

# -----------------------------------------------------------------------------
# MapColumn


class MapColumn(AbstractColumn):
    def __init__(self,  dtype, kwargs):
        assert is_map(dtype)
        super().__init__(dtype)
        self._key_data = _column_constructor(List_(dtype.key_dtype))
        self._item_data = _column_constructor(List_(dtype.item_dtype))
        self.map = MapMethods(self)

    def _invariant(self):
        assert len(self._key_data) == len(self._item_data)
        assert len(self._item_data) == len(self._validity)
        assert 0 <= self._offset and self._offset <= len(self._item_data)
        assert 0 <= self._length and self._offset + \
            self._length <= len(self._item_data)
        rng = range(self._offset, self._offset+self._length)
        assert self.null_count == sum(self._validity[i] for i in rng)

    # implementing abstract methods ----------------------------------------------

    @property
    def ismutable(self):
        """Can this column/frame be extended without side effecting """
        rlengths = self._raw_lengths()
        if len(set(rlengths)) == 1:
            return rlengths[0] == self._offset+self._length
        else:
            return False

    def memory_usage(self, deep=False):
        """Return the memory usage of the Frame (if deep then buffer sizes)."""
        osize = self._offsets.itemsize
        vsize = self._validity.itemsize
        kusage = self._key_data.memory_usage(deep)
        iusage = self._item_data.memory_usage(deep)
        if not deep:
            nchars = (self._offsets[self._offset +
                      self.length]-self._offsets[self._offset])
            return self._length * vsize + self._length * osize + kusage + iusage
        else:
            return len(self._validity)*vsize + len(self.self._offsets)*osize + kusage + iusage

    def append(self, value):
        """Append value to the end of the column/frame"""
        if value is None:
            if self._dtype.nullable:
                self._null_count += 1
                self._validity.append(False)
            else:
                raise TypeError("a map/dict is required (got type NoneType)")
        else:
            self._validity.append(True)
            self._key_data.append(list(value.keys()))
            self._item_data.append(list(value.values()))
        self._length += 1

    def get(self, i, fill_value):
        """Get ith row from column/frame"""
        j = self._offset+i
        if not self._validity[j]:
            return fill_value
        else:
            keys = self._key_data[j]
            items = self._item_data[j]
            return {k: i for k, i in zip(keys, items)}

    def __iter__(self):
        """Return the iterator object itself."""
        for i in range(self._length):
            j = self._offset+i
            if self._validity[j]:
                keys = self._key_data[j]
                items = self._item_data[j]
                yield {k: i for k, i in zip(keys, items)}
            else:
                yield None

    def _copy(self, deep, offset, length):
        if deep:
            res = ListColumn(self.dtype)
            res._length = length
            res._key_data = self._key_data._copy(deep, offset, length)
            res._item_data = self._item_data._copy(deep, offset, length)
            res._validity = self._validity[offset: offset+length]
            res._null_count = sum(res._validity)
            return res
        else:
            return copy.copy(self)

    def _raw_lengths(self):
        return self._key_data._raw_lengths() + self._item_data._raw_lengths()

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


# ------------------------------------------------------------------------------
# registering the factory
_set_column_constructor(is_map, MapColumn)

# -----------------------------------------------------------------------------
# MapMethods


@dataclass
class MapMethods:
    """Vectorized list functions for ListColumn"""
    _parent: MapColumn

    def keys(self):
        me = self._parent
        return me._key_data

    def values(self):
        me = self._parent
        return me._item_data

    def _map_map(self, fun, dtype: Optional[DType] = None):
        me = self._parent
        if dtype is None:
            dtype = me._dtype
        res = _column_constructor(dtype)
        for i in range(me._length):
            j = me._offset+i
            if me._validity[j]:
                res.append(fun(me[j]))
            else:
                res.append(None)
        return res

    def get(self, i, fill_value):
        me = self._parent
        def fun(xs): return xs.get(i, fill_value)
        return self._map_map(fun, me.dtype.item_dtype)

    # def _map_map(self, fun, dtype:Optional[DType]=None):
    #     me = self._parent
    #     if dtype is None:
    #         dtype = me._dtype
    #     res = _column_constructor(dtype)
    #     for i in range(me._length):
    #         j = me._offset+i
    #         if me._validity[j]:
    #             res.append(fun(me[j]))
    #         else:
    #             res.append(None)
    #     return res

    # def map_values(self, fun, dtype:Optional[DType]=None):
    #     func = lambda xs: map(fun,xs)
    #     return self._map_map(func, dtype)


# ops on maps --------------------------------------------------------------
#  'get',
#  'items',
#  'keys',
#  'pop',
#  'popitem',
#  'setdefault',
#  'update',
#  'values'
