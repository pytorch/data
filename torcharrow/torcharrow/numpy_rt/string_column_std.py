import array as ar
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.ma as ma
import torcharrow.dtypes as dt
from tabulate import tabulate
from torcharrow.expression import expression
from torcharrow.istring_column import IStringColumn, IStringMethods
from torcharrow.scope import ColumnFactory


# ------------------------------------------------------------------------------
# StringColumnStd


class StringColumnStd(IStringColumn):

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
    def __init__(self, scope, device, dtype, data, mask):  # REP offsets
        assert dt.is_string(dtype)
        super().__init__(scope, device, dtype)

        self._data = data
        self._mask = mask
        self.str = StringMethodsStd(self)
        # REP: self._offsets = offsets

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(scope, device, dtype, mask=None):
        _mask = mask if mask is not None else ar.array("b")
        # REP  ar.array("I", [0])
        return StringColumnStd(scope, device, dtype, [], _mask)

    @staticmethod
    def _full(scope, device, data, dtype=None, mask=None):
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
        return StringColumnStd(scope, device, dtype, data, mask)

    def _append_null(self):
        self._mask.append(True)
        self._data.append(dt.String.default)
        # REP: offsets.append(offsets[-1])

    def _append_value(self, value):
        assert isinstance(value, str)
        self._mask.append(False)
        self._data.append(value)
        # REP: offsets.append(offsets[-1] + len(i))

    def _append_data(self, value):
        self._data.append(value)

    def _finalize(self):
        self._data = np.array(self._data, dtype=object)
        if isinstance(self._mask, (bool, np.bool8)):
            self._mask = StringColumnStd._valid_mask(len(self._data))
        elif isinstance(self._mask, ar.array):
            self._mask = np.array(self._mask, dtype=np.bool8, copy=False)
        else:
            assert isinstance(self._mask, np.ndarray)
        return self

    def __len__(self):
        return len(self._data)

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        return self._mask[i]

    def getdata(self, i):
        return self._data[i]
        # REP: return self._data[self._offsets[i]: self._offsets[i + 1]]

    @staticmethod
    def _valid_mask(ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def gets(self, indices):
        data = self._data[indices]
        mask = self._mask[indices]
        return self.scope._FullColumn(data, self.dtype, self.device, mask)

    def slice(self, start, stop, step):
        range = slice(start, stop, step)
        return self.scope._FullColumn(
            self._data[range], self.dtype, self.device, self._mask[range]
        )

    def append(self, values):
        """Returns column/dataframe with values appended."""
        tmp = self.scope.Column(values, dtype=self.dtype, device=self.device)
        return self.scope._FullColumn(
            np.append(self._data, tmp._data),
            self.dtype,
            self.device,
            np.append(self._mask, tmp._mask),
        )

    # operators ---------------------------------------------------------------
    @expression
    def __eq__(self, other):
        if isinstance(other, StringColumnStd):
            res = self._EmptyColumn(
                dt.Boolean(self.dtype.nullable or other.dtype.nullable),
                self._mask | other._mask,
            )
            for (m, i), (n, j) in zip(self.items(), other.items()):
                if m or n:
                    res._append_data(dt.Boolean.default)
                else:
                    res._append_data(i == j)
            return res._finalize()
        else:
            res = self._EmptyColumn(dt.Boolean(self.dtype.nullable), self._mask)
            for (m, i) in self.items():
                if m:
                    res._append_data(dt.Boolean.default)
                else:
                    res._append_data(i == other)
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
        typ = f"dtype: {self.dtype}, length: {self.length()}, null_count: {self.null_count()}"
        return tab + dt.NL + typ


# ------------------------------------------------------------------------------
# StringMethodsStd


class StringMethodsStd(IStringMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: StringColumnStd):
        super().__init__(parent)

    def cat(self, others=None, sep: str = "", fill_value: str = None) -> IStringColumn:
        """
        Concatenate strings with given separator and n/a substitition.
        """
        me = cast(StringColumnStd, self._parent)
        assert all(me.device == other.device for other in others)

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


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.String.typecode + "_empty", "std"), StringColumnStd._empty)
ColumnFactory.register((dt.String.typecode + "_full", "std"), StringColumnStd._full)


def _is_not_str(s):
    return not isinstance(s, str)
