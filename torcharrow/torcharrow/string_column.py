import array as ar
import numpy as np
import numpy.ma as ma
from typing import Literal, Iterable

import copy
from dataclasses import dataclass


from .session import ColumnFactory
from .column import AbstractColumn
from .dtypes import NL, Boolean, Field, Int64, List_, String, Struct, is_string, string, typeof_np_ndarray
from .tabulate import tabulate
from .expression import expression

# ------------------------------------------------------------------------------
# StringColumn


class StringColumn(AbstractColumn):

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
    def __init__(self, session, to, dtype, data, mask):  # REP offsets
        assert is_string(dtype)
        super().__init__(session, to, dtype)

        self._data = data
        self._mask = mask
        # REP: self._offsets = offsets

        self.str = StringMethods(self)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(session, to, dtype, mask=None):
        _mask = mask if mask is not None else ar.array("b")
        # REP  ar.array("I", [0])
        return StringColumn(session, to, dtype, [], _mask)

    @staticmethod
    def _full(session, to, data, dtype=None, mask=None):
        assert isinstance(data, np.ndarray) and data.ndim == 1
        if dtype is None:
            dtype = typeof_np_ndarray(data.dtype)
            if dtype is None:  # could be an object array
                mask = np.vectorize(is_not_str)(arr)
                dtype = string
        else:
            pass
            # if dtype != typeof_np_ndarray:
            # pass
            # TODO refine this test
            # raise TypeError(f'type of data {data.dtype} and given type {dtype }must be the same')
        if not is_string(dtype):
            raise TypeError(
                f'construction of columns of type {dtype} not supported')
        if mask is None:
            mask = np.vectorize(_is_not_str)(data)
        elif len(data) != len(mask):
            raise ValueError(
                f'data length {len(data)} must be mask length {len(mask)}')
        # TODO check that all non-masked items are strings
        return StringColumn(session, to, dtype, data, mask)

    def _append_null(self):
        self._mask.append(True)
        self._data.append(String.default)
        # REP: offsets.append(offsets[-1])

    def _append_value(self, value):
        self._mask.append(False)
        self._data.append(value)
        # REP: offsets.append(offsets[-1] + len(i))

    def _append_data(self, value):
        self._data.append(value)

    def _finalize(self):
        self._data = np.array(self._data, dtype=object)
        if isinstance(self._mask, (bool, np.bool_)):
            self._mask = AbstractColumn._valid_mask(len(self._data))
        elif isinstance(self._mask, ar.array):
            self._mask = np.array(self._mask, dtype=np.bool_, copy=False)
        else:
            assert isinstance(self._mask, np.ndarray)
        return self

    @staticmethod
    def _np(session, to, dtype, npdata):
        data = None
        mask = None
        if isinstance(data, ma.array):
            data = npdata.data
            if isinstance(data.mask, (bool, np.bool_)):
                mask = AbstractColumn._valid_mask(len(self._data))
            elif isinstance(data.mask, np.array):
                mask = data.mask
            else:
                raise AssertionError('unexpected case')
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            data = npdata
            mask = AbstractColumn._valid_mask(len(self._data))
        else:
            raise AssertionError('unexpected case')
        return NumericalColumn(session, to, dtype, data, mask)

    def __len__(self):
        return len(self._data)

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        return self._mask[i]

    def getdata(self, i):
        return self._data[i]
        # REP: return self._data[self._offsets[i]: self._offsets[i + 1]]

    def gets(self, indices):
        data = self._data[indices]
        mask = self._mask[indices]
        return StringColumn(*self._meta(), data, mask)

    def slice(self, start, stop, step):
        range = slice(start, stop, step)
        return StringColumn(*self._meta(), self._data[range], self._mask[range])

    def append(self, values):
        """Returns column/dataframe with values appended."""
        tmp = self.session.Column(values, dtype=self.dtype, to=self.to)
        return StringColumn(*self._meta(),
                            np.append(self._data, tmp._data),
                            np.append(self._mask, tmp._mask))

    # operators ---------------------------------------------------------------
    @ expression
    def __eq__(self, other):
        if isinstance(other, StringColumn):
            res = self._Empty(Boolean(
                self.dtype.nullable or other.dtype.nullable),
                self._mask | other._mask)
            for (m, i), (n, j) in zip(self.items(), other.items()):
                if m or n:
                    res._append_data(Boolean.default)
                else:
                    res._append_data(i == j)
            return res._finalize()
        else:
            res = self._Empty(Boolean(self.dtype.nullable), self._mask)
            for (m, i) in self.items():
                if m:
                    res._append_data(Boolean.default)
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
        return tab + NL + typ


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((String.typecode+"_empty", 'test'), StringColumn._empty)
ColumnFactory.register((String.typecode+"_full", 'test'), StringColumn._full)


def _is_not_str(s):
    return not isinstance(s, str)
# ------------------------------------------------------------------------------
# StringMethods


@ dataclass(frozen=True)
class StringMethods:
    """Vectorized string functions for StringColumn"""

    _parent: StringColumn

    def len(self):
        me = self._parent
        res = me._Empty(Int64(me.dtype.nullable), mask=me._mask)
        for m, i in me.items():
            if m:
                res._append_data(Int64.default)
            else:
                res._append_data(len(i))
        return res._finalize()

    def cat(self, others=None, sep: str = "", fill_value: str = None) -> StringColumn:
        """
        Concatenate strings with given separator and n/a substitition.
        """
        me = self._parent
        assert(all(me.to == other.to for other in others))

        _all = [me]+others

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
        res = me._Empty(String(has_nulls))

        for ws in zip(res_filled):
            # will throw if join is applied on null
            res._append_value(sep.join(ws))
        return res._finalize()

    def slice(
        self, start: int = None, stop: int = None, step: int = None
    ) -> StringColumn:
        """Slice substrings from each element in the data or Index."""
        me = self._parent
        res = me._Empty(me.dtype, mask=me._mask)
        for m, i in me.items():
            if m:
                res._append_data(Int64.default)
            else:
                res._append_data(i[start: stop: step])
        return res._finalize()

    def split(self, sep=None, maxsplit=-1, expand=False):
        """Split strings around given separator/delimiter."""
        if not expand:
            return self.split_to_list(sep, maxsplit, direction="left")
        else:
            return self.split_to_column(sep, maxsplit, direction="left")

    def split_to_list(self, sep, maxsplit, direction):
        # cyclic import
        from .list_column import ListColumn

        assert direction in {"left", "right"}

        me = self._parent
        fun = None
        if direction == "left":

            def fun(i):
                return i.split(sep, maxsplit)

        elif direction == "right":

            def fun(i):
                return i.rsplit(sep, maxsplit)

        res = me._Empty(List_(me.dtype), me._mask)
        for m, i in me.items():
            if m:
                res._append_data(List_.default)
            else:
                res._append_data(fun(i))
        return res._finalize()

    def split_to_column(self, sep, maxsplit, direction):
        # cyclic import
        from .dataframe import DataFrame

        assert direction in {"left", "right"}
        assert maxsplit >= 0

        me = self._parent
        res = me._Empty(
            Struct([Field(str(i), String(nullable=True))
                   for i in range(maxsplit + 1)])
        )
        for m, i in me.items():
            if m:
                res._append(tuple([None] * maxsplit))
            else:
                if direction == "left":
                    ws = i.split(sep, maxsplit)
                    ws = ws + ([None] * (maxsplit + 1 - len(ws)))
                elif direction == "right":
                    ws = i.rsplit(sep, maxsplit)
                    ws = ([None] * (maxsplit + 1 - len(ws))) + ws
                else:
                    raise AssertionError(
                        "direction must be in {'left', 'right'}")
                res._append(tuple(ws))
        return res._finalize()

    @ staticmethod
    def _isinteger(s: str):
        try:
            _ = int(s)
            return True
        except ValueError:
            return False

    def isinteger(self):
        """Check whether string forms a positive/negative integer"""
        return self._vectorize_boolean(StringMethods._isinteger)

    @ staticmethod
    def _isfloat(s: str):
        try:
            _ = float(s)
            return True
        except ValueError:
            return False

    def isfloat(self):
        """Check whether string forms a positive/negative floating point number"""
        return self._vectorize_boolean(StringMethods._isfloat)

    def isalnum(self):
        return self._vectorize_boolean(str.isalnum)

    def isalpha(self):
        return self._vectorize_boolean(str.isalpha)

    def isascii(self):
        return self._vectorize_boolean(str.isascii)

    def isdecimal(self):
        return self._vectorize_boolean(str.isdecimal)

    def isdigit(self):
        return self._vectorize_boolean(str.isdigit)

    def isidentifier(self):
        return self._vectorize_boolean(str.isidentifier)

    def islower(self):
        return self._vectorize_boolean(str.islower)

    def isnumeric(self):
        return self._vectorize_boolean(str.isnumeric)

    def isprintable(self):
        return self._vectorize_boolean(str.isprintable)

    def isspace(self):
        return self._vectorize_boolean(str.isspace)

    def istitle(self):
        return self._vectorize_boolean(str.istitle)

    def isupper(self):
        return self._vectorize_boolean(str.isupper)

    def capitalize(self):
        """Convert strings in the data/Index to be capitalized"""
        return self._vectorize_string(str.capitalize)

    def swapcase(self) -> StringColumn:
        """Change each lowercase character to uppercase and vice versa."""
        return self._vectorize_string(str.swapcase)

    def title(self) -> StringColumn:
        return self._vectorize_string(str.title)

    def lower(self) -> StringColumn:
        return self._vectorize_string(str.lower)

    def upper(self) -> StringColumn:
        return self._vectorize_string(str.upper)

    def casefold(self) -> StringColumn:
        return self._vectorize_string(str.casefold)

    def repeat(self, repeats):
        """
        Duplicate each string in the data or Index.
        """
        pass

    def pad(self, width, side="left", fillchar=" "):
        fun = None
        if side == "left":

            def fun(i):
                return i.ljust(width, fillchar)

        if side == "right":

            def fun(i):
                return i.rjust(width, fillchar)

        if side == "center":

            def fun(i):
                return i.center(width, fillchar)

        return self._vectorize_string(fun)

    def ljust(self, width, fillchar=" "):
        def fun(i):
            return i.ljust(width, fillchar)

        return self._vectorize_string(fun)

    def rjust(self, width, fillchar=" "):
        def fun(i):
            return i.rjust(width, fillchar)

        return self._vectorize_string(fun)

    def center(self, width, fillchar=" "):
        def fun(i):
            return i.center(width, fillchar)

        return self._vectorize_string(fun)

    def zfill(self, width):
        def fun(i):
            return i.zfill(width)

        return self._vectorize_string(fun)

    def translate(self, table):
        def fun(i):
            return i.translate(table)

        return self._vectorize_string(fun)

    def count(self, pat, flags=0):
        """Count occurrences of pattern in each string"""
        assert flags == 0
        # TODOL for now just count of FIXED strings, i..e no RE

        def fun(i):
            return i.count(pat)

        return self._vectorize_int64(fun)

    def startswith(self, pat):
        """Test if the beginning of each string element matches a pattern."""

        def pred(i):
            return i.startswith(pat)

        return self._vectorize_boolean(pred)

    def endswith(self, pat):
        """Test if the end of each string element matches a pattern."""

        def pred(i):
            return i.endswith(pat)

        return self._vectorize_boolean(pred)

    def find(self, sub, start=0, end=None):
        def fun(i):
            return i.find(sub, start, end)

        return self._vectorize_int64(fun)

    def rfind(self, sub, start=0, end=None):
        def fun(i):
            return i.rfind(sub, start, end)

        return self._vectorize_int64(fun)

    def index(self, sub, start=0, end=None):
        def fun(i):
            # raises a ValueError when the substring is not found
            return i.index(sub, start, end)

        return self._vectorize_int64(fun)

    def rindex(self, sub, start=0, end=None):
        def fun(i):
            # raises a ValueError when the substring is not found
            return i.rindex(sub, start, end)

        return self._vectorize_int64(fun)

    def _vectorize_boolean(self, pred):
        return self._parent._vectorize(pred, Boolean(self._parent.dtype.nullable))

    def _vectorize_string(self, func):
        return self._parent._vectorize(func, String(self._parent.dtype.nullable))

    def _vectorize_int64(self, func):
        return self._parent._vectorize(func, Int64(self._parent.dtype.nullable))
