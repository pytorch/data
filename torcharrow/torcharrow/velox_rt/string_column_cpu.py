import array as ar
from dataclasses import dataclass
from typing import cast

import _torcharrow as velox
import numpy as np
import numpy.ma as ma
import torcharrow.dtypes as dt
from tabulate import tabulate
from torcharrow.expression import expression
from torcharrow.istring_column import IStringColumn, IStringMethods
from torcharrow.scope import ColumnFactory

from .column import ColumnFromVelox
from .typing import get_velox_type
from .column import ColumnFromVelox

# ------------------------------------------------------------------------------
# StringColumnCpu


class StringColumnCpu(IStringColumn, ColumnFromVelox):

    # Remark: Choosing a representation:
    #
    # Perf: Append and slice via list of string:
    #
    # runtime: type, operations,...
    # 573: array, append, slice, tounicode
    # 874: ndarray(str_), append, memoryview,slice,tobytes, decode(utgf-32)
    # 209: list, append, index
    # 248: ndarray(object), append, index
    # 365: string, concat, slice
    #
    # So we use np.ndarray(object) for now

    # private constructor
    def __init__(self, scope, to, dtype, data, mask):  # REP offsets
        assert dt.is_string(dtype)
        super().__init__(scope, to, dtype)

        self._data = velox.Column(get_velox_type(dtype))
        for m, d in zip(mask.tolist(), data):
            if m:
                self._data.append_null()
            else:
                self._data.append(d)
        self._finialized = False

        self.str = StringMethodsCpu(self)
        # REP: self._offsets = offsets

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(scope, to, dtype, mask=None):
        _mask = mask if mask is not None else ar.array("b")
        # REP  ar.array("I", [0])
        return StringColumnCpu(scope, to, dtype, [], _mask)

    @staticmethod
    def _full(scope, to, data, dtype=None, mask=None):
        assert isinstance(data, np.ndarray) and data.ndim == 1
        if dtype is None:
            dtype = dt.typeof_np_ndarray(data.dtype)
            if dtype is None:  # could be an object array
                mask = np.vectorize(_is_not_str)(data)
                dtype = dt.string
        else:
            pass
            # if dtype != typeof_np_ndarray:
            # pass
            # TODO refine this test
            # raise TypeError(f'type of data {data.dtype} and given type {dtype }must be the same')
        if not dt.is_string(dtype):
            raise TypeError(f"construction of columns of type {dtype} not supported")
        if mask is None:
            mask = np.vectorize(_is_not_str)(data)
        elif len(data) != len(mask):
            raise ValueError(f"data length {len(data)} must be mask length {len(mask)}")
        # TODO check that all non-masked items are strings
        return StringColumnCpu(scope, to, dtype, data, mask)

    def _append_null(self):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        else:
            self._data.append(value)

    def _append_data(self, value):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        self._data.append(value)

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
            return self._data[i]

    @staticmethod
    def _valid_mask(ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def gets(self, indices):
        data = self._data[indices]
        mask = self._mask[indices]
        return self.scope._FullColumn(data, self.dtype, self.to, mask)

    def slice(self, start, stop, step):
        range = slice(start, stop, step)
        return self.scope._FullColumn(
            self._data[range], self.dtype, self.to, self._mask[range]
        )

    # operators ---------------------------------------------------------------
    @expression
    def __eq__(self, other):
        if isinstance(other, StringColumnCpu):
            res = self._EmptyColumn(
                dt.Boolean(self.dtype.nullable or other.dtype.nullable),
            )
            for (m, i), (n, j) in zip(self.items(), other.items()):
                if m or n:
                    res._data.append_null()
                else:
                    res._data.append(i == j)
            return res._finalize()
        else:
            res = self._EmptyColumn(dt.Boolean(self.dtype.nullable))
            for (m, i) in self.items():
                if m:
                    res._data.append_null()
                else:
                    res._data.append(i == other)
            return res._finalize()

    # printing ----------------------------------------------------------------

    def __str__(self):
        def quote(x):
            return f"'{x}'"

        return f"Column([{', '.join('None' if i is None else quote(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate(
            [["None" if i is None else f"'{i}'"] for i in self],
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self.dtype}, length: {self.length()}, null_count: {self.null_count()}, device: cpu"
        return tab + dt.NL + typ


# ------------------------------------------------------------------------------
# StringMethodsCpu


class StringMethodsCpu(IStringMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: StringColumnCpu):
        super().__init__(parent)

    def cat(self, others=None, sep: str = "", fill_value: str = None) -> IStringColumn:
        """
        Concatenate strings with given separator and n/a substitition.
        """
        me = cast(StringColumnCpu, self._parent)
        assert all(me.to == other.to for other in others)

        _all = [me] + others

        # mask
        res_mask = me._mask
        if fill_value is None:
            for one in _all:
                if res_mask is None:
                    res_mak = one.mask
                elif one.mask is not None:
                    res_mask = res_mask | one.mask

        # fill
        res_filled = []
        for one in _all:
            if fill_value is None:
                res_filled.append(one.fillna(fill_value))
            else:
                res_filled.append(one)
        # join
        has_nulls = fill_value is None and any(one.nullable for one in _all)
        res = me._EmptyColumn(dt.String(has_nulls))

        for ws in zip(res_filled):
            # will throw if join is applied on null
            res._append_value(sep.join(ws))
        return res._finalize()

    def lower(self) -> IStringColumn:
        return ColumnFromVelox.from_velox(self._parent.scope, self._parent.to, self._parent.dtype, self._parent._data.lower(), True)

    def upper(self) -> IStringColumn:
        return ColumnFromVelox.from_velox(self._parent.scope, self._parent.to, self._parent.dtype, self._parent._data.upper(), True)

    def isalpha(self) -> IStringColumn:
        return ColumnFromVelox.from_velox(self._parent.scope, self._parent.to, dt.Boolean(self._parent.dtype.nullable), self._parent._data.isalpha(), True)


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.String.typecode + "_empty", "cpu"), StringColumnCpu._empty)
ColumnFactory.register((dt.String.typecode + "_full", "cpu"), StringColumnCpu._full)


def _is_not_str(s):
    return not isinstance(s, str)
