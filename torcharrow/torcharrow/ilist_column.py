import abc
import array as ar
import builtins
import functools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torcharrow.dtypes as dt

from .icolumn import IColumn
from .scope import ColumnFactory

# -----------------------------------------------------------------------------
# IListColumn


class IListColumn(IColumn):

    # private constructor
    def __init__(self, scope, device, dtype):
        assert dt.is_list(dtype)
        super().__init__(scope, device, dtype)
        self.list = IListMethods(self)


# -----------------------------------------------------------------------------
# IListMethods


class IListMethods(abc.ABC):
    """Vectorized list functions for IListColumn"""

    def __init__(self, parent):
        self._parent: IListColumn = parent

    def length(self):
        me = self._parent
        return me._vectorize(len, dt.Int64(me.dtype.nullable))

    def join(self, sep):
        """Join lists contained as elements with passed delimiter."""
        me = self._parent
        assert dt.is_string(me.dtype.item_dtype)

        def fun(i):
            return sep.join(i)

        return me._vectorize(
            fun, dt.String(me.dtype.item_dtype.nullable or me.dtype.nullable)
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

        return me._vectorize(fun, dt.Int64(me.dtype.nullable))

    def map(self, fun, dtype: Optional[dt.DType] = None):
        me = self._parent

        def func(xs):
            return list(builtins.map(fun, xs))

        if dtype is None:
            dtype = me.dtype
        return me._vectorize(func, dtype)

    def filter(self, pred):
        print()
        me = self._parent

        def func(xs):
            return list(builtins.filter(pred, xs))

        return me._vectorize(func, me.dtype)

    def reduce(self, fun, initializer=None, dtype: Optional[dt.DType] = None):
        me = self._parent

        def func(xs):
            return functools.reduce(fun, xs, initializer)

        dtype = me.dtype.item_dtype if dtype is None else dtype
        return me._vectorize(func, dtype)

    def flatmap(self, fun, dtype: Optional[dt.DType] = None):
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
