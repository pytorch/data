import array as ar
from dataclasses import dataclass
import abc
import typing as ty

# TODO: use re2
import re
#import re2 as re  # type: ignore

import numpy as np
import numpy.ma as ma

import torcharrow.dtypes as dt
from torcharrow.expression import Call

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
        me = self._parent

        if not expand:

            def fun(i):
                return i.split(sep, maxsplit)

            return self._vectorize_list_string(fun)
        else:
            if maxsplit < 1:
                raise ValueError("maxsplit must be >0")

            def fun(i):
                ws = i.split(sep, maxsplit)
                return tuple(ws + ([None] * (maxsplit + 1 - len(ws))))

            dtype = dt.Struct(
                [dt.Field(str(i), dt.String(nullable=True)) for i in range(maxsplit + 1)],
                nullable=me.dtype.nullable,
            )

            return me._vectorize(fun, dtype=dtype)

    def rsplit(self, sep=None, maxsplit=-1, expand=False):
        """Split strings around given separator/delimiter."""
        me = self._parent

        if not expand:

            def fun(i):
                return i.rsplit(sep, maxsplit)

            return self._vectorize_list_string(fun)
        else:
            if maxsplit < 1:
                raise ValueError("maxsplit must be >0")

            def fun(i):
                ws = i.rsplit(sep, maxsplit)
                return tuple(
                    ([None] * (maxsplit + 1 - len(ws))) + i.rsplit(sep, maxsplit)
                )

            dtype = dt.Struct(
                [dt.Field(str(i), dt.String(nullable=True)) for i in range(maxsplit + 1)],
                nullable=me.dtype.nullable,
            )

            return me._vectorize(fun, dtype=dtype)

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

    # helper -----------------------------------------------------

    def _vectorize_boolean(self, pred):
        return self._parent._vectorize(pred, dt.Boolean(self._parent.dtype.nullable))

    def _vectorize_string(self, func):
        return self._parent._vectorize(func, dt.String(self._parent.dtype.nullable))

    def _vectorize_list_string(self, func):
        return self._parent._vectorize(
            func, dt.List(dt.string, self._parent.dtype.nullable)
        )

    def _vectorize_int64(self, func):
        return self._parent._vectorize(func, dt.Int64(self._parent.dtype.nullable))

    # Regular expressions -----------------------------------------------------

    def count_re(
        self,
        pattern: ty.Union[str, re.Pattern]
        # flags: int = 0, not supported
    ):
        """Count occurrences of pattern in each string"""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        def func(text):
            return len(re.findall(pattern, text))

        return self._vectorize_int64(func)

    def match_re(self, pattern: ty.Union[str, re.Pattern]):
        """Determine if each string matches a regular expression (see re.match())"""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        def func(text):
            return True if pattern.match(text) else False

        return self._vectorize_boolean(func)

    def replace_re(
        self,
        pattern: ty.Union[str, re.Pattern],
        repl: ty.Union[str, ty.Callable],
        count=0,
        # flags = 0
    ):
        """Replace for each item the search string or pattern with the given value"""

        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        def func(text):
            return re.sub(pattern, repl, text, count)

        return self._vectorize_string(func)

    def contains_re(
        self,
        pattern: ty.Union[str, re.Pattern],
    ):
        """Test for each item if pattern is contained within a string; returns a boolean"""

        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        def func(text):
            return pattern.search(text) is not None

        return self._vectorize_boolean(func)

    def findall_re(self, pattern: ty.Union[str, re.Pattern]):
        """
        Find for each item all occurrences of pattern (see re.findall())
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        def func(text):
            return pattern.findall(text)

        return self._vectorize_list_string(func)

    def extract_re(self, pattern: ty.Union[str, re.Pattern]):
        """Return capture groups in the regex as columns of a dataframe"""
        # generalizes Pandas extract ad extractall;
        # always Pandas' expand = True
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        num_groups = pattern.groups
        if num_groups == 0:
            raise ValueError("pattern contains no capture groups")
        group_names = pattern.groupindex
        inverted_group_names = {v - 1: k for k, v in group_names.items()}
        columns = []
        for i in range(num_groups):
            if i in inverted_group_names:
                columns.append(inverted_group_names[i])
            else:
                columns.append(str(i))
        dtype = dt.Struct([dt.Field(c, dt.String(nullable=True)) for c in columns])

        def func(text):
            gps = pattern.search(text)
            if gps is None:
                # TODO decide: return all groups as none or have just one none
                return tuple([None] * num_groups)
            return gps.groups()

        return self._parent._vectorize(func, dtype)

    def split_re(self, pattern: ty.Union[str, re.Pattern], maxsplit=-1, expand=False):
        """Split each string from the beginning (see re.split)
        returning them as a list or dataframe (expand=True)"""
        me = self._parent

        # Python's re module will not split the string if maxsplit<0
        maxsplit = max(maxsplit, 0)

        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        if not expand:

            def fun(text):
                return pattern.split(text, maxsplit)

            return self._vectorize_list_string(fun)
        else:
            if maxsplit < 1:
                raise ValueError("maxsplit must be >0")

            def fun(text):
                ws = pattern.split(text, maxsplit)
                return tuple(ws + ([None] * (maxsplit + 1 - len(ws))))

            dtype = dt.Struct(
                [dt.Field(str(i), dt.String(nullable=True)) for i in range(maxsplit + 1)],
                nullable=me.dtype.nullable,
            )

            return self._parent._vectorize(fun, dtype=dtype)
