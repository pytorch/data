import array as ar
import numpy as np
import numpy.ma as ma

import builtins
import copy
import functools
from dataclasses import dataclass
from typing import List, Optional

from .session import ColumnFactory
from .column import AbstractColumn
from .dtypes import NL, Boolean, DType, Int64, List_, String, Uint32, is_list, is_string
from .tabulate import tabulate

# -----------------------------------------------------------------------------
# ListColumn


class ListColumn(AbstractColumn):

    # private constructor
    def __init__(self, session, to, dtype, data, offsets, mask):
        assert is_list(dtype)
        super().__init__(session, to, dtype)

        self._data = data
        self._offsets = offsets
        self._mask = mask

        self.list = ListMethods(self)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time
    @staticmethod
    def _empty(session, to, dtype, mask=None):
        _mask = mask if mask is not None else ar.array("b")
        return ListColumn(session, to, dtype, session._Empty(dtype.item_dtype, to), ar.array("I", [0]), _mask)

    def _append_null(self):
        self._mask.append(True)
        self._offsets.append(self._offsets[-1])
        self._data._extend([])

    def _append_value(self, value):
        self._mask.append(False)
        self._offsets.append(self._offsets[-1] + len(value))
        self._data._extend(value)

    def _append_data(self, data):
        self._offsets.append(self._offsets[-1] + len(data))
        self._data._extend(data)

    def _finalize(self):
        self._data = self._data._finalize()
        self._offsets = np.array(self._offsets, dtype=np.int32, copy=False)
        if not isinstance(self._mask, np.ndarray):
            self._mask = np.array(self._mask, dtype=np.bool_, copy=False)
        return self

    def __len__(self):
        return len(self._offsets)-1

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        return self._mask[i]

    def getdata(self, i):
        return list(self._data[self._offsets[i]: self._offsets[i + 1]])

    def append(self, values):
        """Returns column/dataframe with values appended."""
        # tmp = self.session.Column(values, dtype=self.dtype, to = self.to)
        # res= ListColumn(*self._meta(),
        #     np.append(self._data,tmp._data),
        #     np.append(self._offsets,tmp._offsets[1:] + self._offsets[-1]),
        #     np.append(self._mask,tmp._mask))

        # TODO replace this with vectorized code like the one above, except that is buggy
        res = self.session._Empty(self.dtype)
        for v in self:
            res._append(v)
        for v in values:
            res._append(v)
        return res._finalize()

    def concat(self, values):
        """Returns column/dataframe with values appended."""
        # tmp = self.session.Column(values, dtype=self.dtype, to = self.to)
        # res= ListColumn(*self._meta(),
        #     np.append(self._data,tmp._data),
        #     np.append(self._offsets,tmp._offsets[1:] + self._offsets[-1]),
        #     np.append(self._mask,tmp._mask))

        # TODO replace this with vectorized code like the one above, except that is buggy
        res = self.session._Empty(self.dtype)
        for v in self:
            res._append(v)
        for v in values:
            res._append(v)
        return res._finalize()

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
ColumnFactory.register((List_.typecode+"_empty", 'test'), ListColumn._empty)


# -----------------------------------------------------------------------------
# ListMethods


@dataclass(frozen=True)
class ListMethods:
    """Vectorized list functions for ListColumn"""

    _parent: ListColumn

    def join(self, sep):
        """Join lists contained as elements with passed delimiter."""
        me = self._parent
        assert is_string(me.dtype.item_dtype)

        def fun(i):
            return sep.join(i)

        return me._vectorize(
            fun, String(me.dtype.item_dtype.nullable or me.dtype.nullable)
        )

    def get(self, i):
        me = self._parent

        def fun(xs):
            return xs[i]

        return me._vectorize(fun, me.dtype.item_dtype.with_null(me.dtype.nullable))

    def count(self, elem, flags=0):
        me = self._parent

        def fun(i):
            return i.count(elem)

        return me._vectorize(fun, Int64(me.dtype.nullable))

    def map(self, fun, dtype: Optional[DType] = None):
        me = self._parent

        def func(xs):
            return list(builtins.map(fun, xs))

        if dtype is None:
            dtype = me.dtype
        return me._vectorize(func, dtype)

    def filter(self, pred):
        me = self._parent

        def func(xs):
            return list(builtins.filter(pred, xs))

        return me._vectorize(func, me.dtype)

    def reduce(self, fun, initializer=None, dtype: Optional[DType] = None):
        me = self._parent

        def func(xs):
            return functools.reduce(fun, xs, initializer)

        dtype = me.dtype.item_dtype if dtype is None else dtype
        return me._vectorize(func, dtype)

    def flatmap(self, fun, dtype: Optional[DType] = None):
        # dtype must be given, if result is different from argument column
        me = self._parent

        def func(xs):
            ys = []
            for x in xs:
                ys._extend(fun(x))
            return ys

        return me._vectorize(func, me.dtype)


# ops on list  --------------------------------------------------------------
#  'count',
#  'extend',
#  'index',
#  'insert',
#  'pop',
#  'remove',
#  'reverse',
