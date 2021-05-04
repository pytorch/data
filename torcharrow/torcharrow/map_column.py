import array as ar
import copy
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from .session import ColumnFactory
from .column import AbstractColumn
from .dtypes import NL, DType, List_, is_map, Map
from .list_column import ListColumn
from .tabulate import tabulate

# -----------------------------------------------------------------------------
# MapColumn


class MapColumn(AbstractColumn):

    def __init__(self, session, to, dtype, key_data, item_data, mask):
        assert is_map(dtype)
        super().__init__(session, to, dtype)

        self._key_data = key_data
        self._item_data = item_data
        self._mask = mask

        self.map = MapMethods(self)

    # Lifecycle: _empty -> _append* -> _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(session, to, dtype, mask=None):
        key_data = session._Empty(
            List_(dtype.key_dtype).with_null(dtype.nullable))
        item_data = session._Empty(
            List_(dtype.item_dtype).with_null(dtype.nullable))
        _mask = mask if mask is not None else ar.array("b")
        return MapColumn(session, to, dtype, key_data, item_data, _mask)

    def _append_null(self):
        self._mask.append(True)
        self._key_data._append_null()
        self._item_data._append_null()

    def _append_value(self, value):
        self._mask.append(False)
        self._key_data._append_value(list(value.keys()))
        self._item_data._append_value(list(value.values()))

    def _append_data(self, value):
        self._key_data._append_value(list(value.keys()))
        self._item_data._append_value(list(value.values()))

    def _finalize(self, mask=None):
        self._key_data = self._key_data._finalize()
        self._item_data = self._item_data._finalize()
        if not isinstance(self._mask, np.ndarray):
            self._mask = np.array(self._mask, dtype=np.bool_, copy=False)
        return self

    def __len__(self):
        return len(self._key_data)

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        return self._mask[i]

    def getdata(self, i):
        return {k: v for k, v in zip(self._key_data[i], self._item_data[i])}

    def append(self, values):
        """Returns column/dataframe with values appended."""
        tmp = self.session.Column(values, dtype=self.dtype, to=self.to)
        return MapColumn(*self._meta(),
                         self._key_data.append(tmp._key_data),
                         self._item_data.append(tmp._item_data),
                         np.append(self._mask, tmp._mask))

    # printing ----------------------------------------------------------------
    def __str__(self):
        return f"Column([{', '.join('None' if i is None else str(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate(
            [["None" if i is None else str(i)] for i in self],
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self._dtype}, length: {self.length()}, null_count: {self.null_count()}"
        return tab + NL + typ


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((Map.typecode+"_empty", 'test'), MapColumn._empty)

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

    def get(self, i, fill_value):
        me = self._parent

        def fun(xs):
            # TODO improve perf by looking at lists instead of first building a map
            return xs.get(i, fill_value)

        return me._vectorize(fun, me.dtype.item_dtype)


# ops on maps --------------------------------------------------------------
#  'get',
#  'items',
#  'keys',
#  'pop',
#  'popitem',
#  'setdefault',
#  'update',
#  'values'
