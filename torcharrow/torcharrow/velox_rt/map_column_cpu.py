import array as ar
import copy
from collections import OrderedDict
from dataclasses import dataclass

import _torcharrow as velox
import numpy as np
import torcharrow.dtypes as dt
from tabulate import tabulate
from torcharrow.icolumn import IColumn
from torcharrow.imap_column import IMapColumn, IMapMethods
from torcharrow.scope import ColumnFactory

from .column import ColumnFromVelox
from .typing import get_velox_type

# -----------------------------------------------------------------------------
# IMapColumn


class MapColumnCpu(IMapColumn, ColumnFromVelox):
    def __init__(self, scope, to, dtype, key_data, item_data, mask):
        assert dt.is_map(dtype)
        super().__init__(scope, to, dtype)

        self._data = velox.Column(
            velox.MAP(get_velox_type(dtype.key_dtype), get_velox_type(dtype.item_dtype))
        )
        # key_col = velox.Column(get_velox_type(dtype.key_dtype))
        # value_col = velox.Column(get_velox_type(dtype.item_dtype))

        self._finialized = False

        self.map = MapMethodsCpu(self)

    # Lifecycle: _empty -> _append* -> _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(scope, to, dtype, mask=None):
        key_data = scope._EmptyColumn(
            dt.List(dtype.key_dtype).with_null(dtype.nullable)
        )
        item_data = scope._EmptyColumn(
            dt.List(dtype.item_dtype).with_null(dtype.nullable)
        )
        _mask = mask if mask is not None else ar.array("b")
        return MapColumnCpu(scope, to, dtype, key_data, item_data, _mask)

    @staticmethod
    def _full(scope, to, data, dtype=None, mask=None):
        assert isinstance(data, tuple) and len(data) == 2
        key_data, item_data = data
        assert isinstance(key_data, IColumn)
        assert isinstance(item_data, IColumn)
        assert len(item_data) == len(key_data)

        if dtype is None:
            dtype = dt.Map(
                dt.typeof_np_ndarray(key_data.dtype),
                dt.typeof_np_ndarray(item_data.dtype),
            )
        # else:
        #     if dtype != dt.typeof_np_dtype(data.dtype):
        #         # TODO fix nullability
        #         # raise TypeError(f'type of data {data.dtype} and given type {dtype} must be the same')
        #         pass
        if not dt.is_map(dtype):
            raise TypeError(f"construction of columns of type {dtype} not supported")
        if mask is None:
            mask = IMapColumn._valid_mask(len(key_data))
        elif len(key_data) != len(mask):
            raise ValueError(
                f"data length {len(key_data)} must be the same as mask length {len(mask)}"
            )
        # TODO check that all non-masked items are legal numbers (i.e not nan)
        return MapColumnCpu(scope, to, dtype, key_data, item_data, mask)

    def _append_null(self):
        raise NotImplementedError()

    def _append_value(self, value):
        if value is None:
            raise NotImplementedError()
        else:
            new_key = self.scope.Column(self._dtype.key_dtype)
            new_value = self.scope.Column(self._dtype.item_dtype)
            new_key = new_key.append(value.keys())
            new_value = new_value.append(value.values())
            self._data.append(new_key._data, new_value._data)

    def _append_data(self, value):
        self._append_value(value)

    def _finalize(self, mask=None):
        self._finialized = True
        return self

    def __len__(self):
        return len(self._data)

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        if i < 0:
            i += len(self._data)
        return self._data.is_null_at(i)

    def getdata(self, i):
        if i < 0:
            i += len(self._data)
        if self._data.is_null_at(i):
            return self.dtype.default
        else:
            key_col = ColumnFromVelox.from_velox(
                self.scope, self._dtype.key_dtype, self._data.keys()[i], True
            )
            value_col = ColumnFromVelox.from_velox(
                self.scope,
                self._dtype.item_dtype,
                self._data.values()[i],
                True,
            )

            return {key_col[j]: value_col[j] for j in range(len(key_col))}

    @staticmethod
    def _valid_mask(ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def append(self, values):
        """Returns column/dataframe with values appended."""
        for value in values:
            if value is None:
                raise NotImplementedError()
            else:
                new_key = self.scope.Column(self._dtype.key_dtype)
                new_value = self.scope.Column(self._dtype.item_dtype)
                new_key = new_key.append(value.keys())
                new_value = new_value.append(value.values())
                self._data.append(new_key._data, new_value._data)
        return self

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
        return tab + dt.NL + typ


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.Map.typecode + "_empty", "cpu"), MapColumnCpu._empty)
# registering the factory
ColumnFactory.register((dt.Map.typecode + "_full", "cpu"), MapColumnCpu._full)
# -----------------------------------------------------------------------------
# MapMethods


@dataclass
class MapMethodsCpu(IMapMethods):
    """Vectorized list functions for IListColumn"""

    def __init__(self, parent: MapColumnCpu):
        super().__init__(parent)

    def keys(self):
        me = self._parent
        return ColumnFromVelox.from_velox(
            me.scope, dt.List(me._dtype.key_dtype), me._data.keys(), True
        )

    def values(self):
        me = self._parent
        return ColumnFromVelox.from_velox(
            me.scope, dt.List(me._dtype.item_dtype), me._data.values(), True
        )


# ops on maps --------------------------------------------------------------
#  'get',
#  'items',
#  'keys',
#  'pop',
#  'popitem',
#  'setdefault',
#  'update',
#  'values'
