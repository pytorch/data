import array as ar
from dataclasses import dataclass
import abc
import numpy as np
import numpy.ma as ma

import torcharrow.dtypes as dt

from .icolumn import IColumn
from .expression import expression
from .scope import ColumnFactory

# ------------------------------------------------------------------------------
# IStringColumn


class IStringColumn(IColumn):

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
    def __init__(self, scope, to, dtype):  # REP offsets
        assert dt.is_string(dtype)
        super().__init__(scope, to, dtype)
        # must be set by subclass
        self.str: IStringMethods = None


# ------------------------------------------------------------------------------
# IStringMethods


class IStringMethods(abc.ABC):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent):
        self._parent: IStringColumn = parent

    def length(self):
        return self._vectorize_int64(len)

    @abc.abstractmethod
    def cat(self, others=None, sep: str = "", fill_value: str = None) -> IStringColumn:
        """
        Concatenate strings with given separator and n/a substitition.
        """
        pass

    def slice(
        self, start: int = None, stop: int = None, step: int = None
    ) -> IStringColumn:
        """Slice substrings from each element in the data or Index."""

        def func(i):
            return i[start:stop:step]

        return self._vectorize_string(func)

    def split(self, sep=None, maxsplit=-1, expand=False):
        """Split strings around given separator/delimiter."""
        if not expand:
            return self.split_to_list(sep, maxsplit, direction="left")
        else:
            return self.split_to_column(sep, maxsplit, direction="left")

    def split_to_list(self, sep, maxsplit, direction):
        # cyclic import
        from .ilist_column import IListColumn

        assert direction in {"left", "right"}

        me = self._parent
        fun = None
        if direction == "left":

            def fun(i):
                return i.split(sep, maxsplit)

        elif direction == "right":

            def fun(i):
                return i.rsplit(sep, maxsplit)

        res = me._EmptyColumn(dt.List(me.dtype), me._mask)
        for m, i in me.items():
            if m:
                res._append_data(dt.List.default)
            else:
                res._append_data(fun(i))
        return res._finalize()

    def split_to_column(self, sep, maxsplit, direction):
        # cyclic import
        from .idataframe import DataFrame

        assert direction in {"left", "right"}
        assert maxsplit >= 0

        me = self._parent
        res = me._EmptyColumn(
            dt.Struct(
                [
                    dt.Field(str(i), dt.String(nullable=True))
                    for i in range(maxsplit + 1)
                ]
            )
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
                    raise AssertionError("direction must be in {'left', 'right'}")
                res._append(tuple(ws))
        return res._finalize()

    @staticmethod
    def _isinteger(s: str):
        try:
            _ = int(s)
            return True
        except ValueError:
            return False

    def isinteger(self):
        """Check whether string forms a positive/negative integer"""
        return self._vectorize_boolean(IStringMethods._isinteger)

    @staticmethod
    def _isfloat(s: str):
        try:
            _ = float(s)
            return True
        except ValueError:
            return False

    def isfloat(self):
        """Check whether string forms a positive/negative floating point number"""
        return self._vectorize_boolean(IStringMethods._isfloat)

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

    def swapcase(self) -> IStringColumn:
        """Change each lowercase character to uppercase and vice versa."""
        return self._vectorize_string(str.swapcase)

    def title(self) -> IStringColumn:
        return self._vectorize_string(str.title)

    def lower(self) -> IStringColumn:
        return self._vectorize_string(str.lower)

    def upper(self) -> IStringColumn:
        return self._vectorize_string(str.upper)

    def casefold(self) -> IStringColumn:
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
        return self._parent._vectorize(pred, dt.Boolean(self._parent.dtype.nullable))

    def _vectorize_string(self, func):
        return self._parent._vectorize(func, dt.String(self._parent.dtype.nullable))

    def _vectorize_int64(self, func):
        return self._parent._vectorize(func, dt.Int64(self._parent.dtype.nullable))
