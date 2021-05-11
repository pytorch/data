import abc
from dataclasses import dataclass

import torcharrow.dtypes as dt

from .icolumn import IColumn

# -----------------------------------------------------------------------------
# IMapColumn


class IMapColumn(IColumn):
    def __init__(self, scope, to, dtype):
        assert dt.is_map(dtype)
        super().__init__(scope, to, dtype)
        # must be set by subclasses
        self.map: IMapMethods = None


# -----------------------------------------------------------------------------
# MapMethods


class IMapMethods(abc.ABC):
    """Vectorized list functions for IListColumn"""

    def __init__(self, parent):
        self._parent: IMapColumn = parent

    @abc.abstractmethod
    def keys(self):
        pass

    @abc.abstractmethod
    def values(self):
        pass

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
