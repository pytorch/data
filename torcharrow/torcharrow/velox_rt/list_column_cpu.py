import array as ar

import _torcharrow as velox
import numpy as np
import torcharrow.dtypes as dt
from tabulate import tabulate
from torcharrow.ilist_column import IListColumn, IListMethods
from torcharrow.scope import ColumnFactory

from .column import ColumnFromVelox
from .typing import get_velox_type

# -----------------------------------------------------------------------------
# IListColumn


class ListColumnCpu(IListColumn, ColumnFromVelox):

    # private constructor
    def __init__(self, scope, device, dtype, data, offsets, mask):
        assert dt.is_list(dtype)
        super().__init__(scope, device, dtype)

        self._data = velox.Column(velox.VeloxArrayType(get_velox_type(dtype.item_dtype)))
        if len(data) > 0:
            self.append(data)
        self._finialized = False

        self.list = ListMethodsCpu(self)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time
    @staticmethod
    def _empty(scope, device, dtype, mask=None):
        _mask = mask if mask is not None else ar.array("b")
        return ListColumnCpu(
            scope,
            device,
            dtype,
            scope._EmptyColumn(dtype.item_dtype, device),
            ar.array("I", [0]),
            _mask,
        )

    def _append_null(self):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        elif value is None:
            self._data.append_null()
        else:
            new_element_column = self._scope.Column(self._dtype.item_dtype)
            new_element_column = new_element_column.append(value)
            self._data.append(new_element_column._data)

    def _append_data(self, data):
        self._append_value(data)

    def _finalize(self):
        self._finialized = True
        return self

    def __len__(self):
        return len(self._data)

    def null_count(self):
        return self._data.get_null_count()

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
            return list(
                ColumnFromVelox.from_velox(
                    self.scope, self.device, self._dtype.item_dtype, self._data[i], False
                )
            )

    def concat(self, values):
        """Returns column/dataframe with values appended."""
        # tmp = self.scope.Column(values, dtype=self.dtype, device = self.device)
        # res= IListColumn(*self._meta(),
        #     np.append(self._data,tmp._data),
        #     np.append(self._offsets,tmp._offsets[1:] + self._offsets[-1]),
        #     np.append(self._mask,tmp._mask))

        # TODO replace this with vectorized code like the one above, except that is buggy
        res = self._EmptyColumn(self.dtype)
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
        return tab + dt.NL + typ

    def __iter__(self):
        """Return the iterator object itself."""
        for i in range(len(self)):
            item = self.get(i)
            if item is None:
                yield item
            else:
                yield list(item)


# ------------------------------------------------------------------------------
# ListMethodsCpu


class ListMethodsCpu(IListMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: ListColumnCpu):
        super().__init__(parent)


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.List.typecode + "_empty", "cpu"), ListColumnCpu._empty)
