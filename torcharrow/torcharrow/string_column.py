import array as ar
import copy
from dataclasses import dataclass
from typing import Literal

from .column import AbstractColumn, Column, _set_column_constructor
from .dtypes import (NL, Boolean, Field, Int64, List_, String, Struct,
                     is_string, string)
from .numerical_column import BooleanColumn, NumericalColumn
from .tabulate import tabulate

# ------------------------------------------------------------------------------
# StringColumn


class StringColumn(AbstractColumn):
    # private

    def __init__(self,  dtype, kwargs=None):
        assert is_string(dtype)
        super().__init__(dtype)
        self._offsets = ar.array('I', [0])  # Uint32
        self._data = ar.array('u')
        self.str = StringMethods(self)

    def _invariant(self):
        assert len(self._data) == len(self._validity)
        assert len(self._data) == len(self._offsets)-1
        assert self.offsets[-1] == len(self._data)
        assert all(self.offset[i] <= self.offset[i+1]
                   for i in range(0, len(self.offsets)+1))
        assert 0 <= self._offset and self._offset <= len(self._data)
        assert 0 <= self._length and self._offset + \
            self._length <= len(self._data)
        rng = range(self._offset, self._offset+self._length)
        assert self.null_count == sum(self.validity[i] for i in rng)

    # implementing abstract methods ----------------------------------------------

    def _raw_lengths(self):
        return [len(self._data)]

    @property
    def ismutable(self):
        """Can this column/frame be extended without side effecting """
        return self._raw_lengths()[0] == self._offsets[self._offset + self._length]

    def memory_usage(self, deep=False):
        """The mimimal memory usage in bytes of the column/frame; if deep then memory usage of referenced buffers."""
        osize = self._offsets.itemsize
        vsize = self._validity.itemsize
        dsize = self._data.itemsize
        if not deep:
            nchars = (self._offsets[self._offset +
                      self.length]-self._offsets[self._offset])
            return self._length * vsize + self._length * osize + nchars * dsize
        else:
            return len(self._validity)*vsize + len(self.self._offsets)*osize + len(self._data)*dsize

    def append(self, cs):
        """Append value to the end of the column/frame"""
        if cs is None:
            if self._dtype.nullable:
                self._null_count += 1
                self._validity.append(False)
                self._offsets.append(self._offsets[-1])
            else:
                raise TypeError("a string is required (got type NoneType)")
        else:
            self._validity.append(True)
            self._data.extend(cs)
            self._offsets.append(self._offsets[-1]+len(cs))
        self._length += 1

    def get(self, i, fill_value):
        """Get item from column/frame for given integer index"""
        j = self._offset+i
        if self._null_count == 0:
            return self._data[self._offsets[j]:self._offsets[j+1]].tounicode()
        elif not self._validity[j]:
            return fill_value
        else:
            return self._data[self._offsets[j]:self._offsets[j+1]].tounicode()

    def __iter__(self):
        """Return the iterator object itself."""
        for i in range(self._length):
            j = self._offset+i
            if self._validity[j]:
                yield self._data[self._offsets[j]:self._offsets[j+1]].tounicode()
            else:
                yield None

    def _copy(self, deep, offset, length):
        if deep:
            res = StringColumn(self.dtype)
            res._length = length
            res._data = self._data[self._offsets[self._offset]: self._offsets[offset+length]]
            res._validity = self._validity[offset: offset+length]
            res._null_count = sum(res._validity)
            return res
        else:
            return copy.copy(self)

    # printing ----------------------------------------------------------------
    def __str__(self):
        return f"Column([{', '.join('None' if i is None else i for i in self)}])"

    
    def __repr__(self):
        tab = tabulate([['None' if i is None else f"'{i}'"]
                       for i in self], tablefmt='plain', showindex=True)
        typ = f"dtype: {self._dtype}, length: {self._length}, null_count: {self._null_count}"
        return tab+NL+typ

    def show_details(self):
        return _Repr(self)


@dataclass
class _Repr:
    parent: StringColumn

    def __repr__(self):
        raise NotImplementedError()


# ------------------------------------------------------------------------------
# registering the factory
_set_column_constructor(is_string, StringColumn)

# ------------------------------------------------------------------------------
# StringMethods


@dataclass(frozen=True)
class StringMethods:
    """Vectorized string functions for StringColumn"""
    _parent: StringColumn

    def len(self):
        """Computes the length of each string element."""
        data = self._parent
        res = NumericalColumn(Int64(data.dtype.nullable))
        for i in range(data._length):
            j = data._offset+i
            # shoud be optimized...
            if data._validity[j]:
                res.append(data._offsets[j+1]-data._offsets[j])
            else:
                res.append(None)
        return res

    # CUDF only
    # def cat(self, others=None, sep:str="", na_rep:str=None):
    #     """
    #     Concatenate strings in the data/Index with given separator.
    #     """
    #     data = self._parent
    #     # result is a string
    #     if others is None and na_rep is None:
    #         return sep.join(s for s in data._iter_drop_na())
    #     if others is None and na_rep is not None:
    #         return sep.join(s for s in data._iter_fill_na(na_rep))

    #     # result is a vector
    #     # TODO should be generalized to list of list
    #     assert isinstance(others, (StringColumn, list)) and len(others) == len(data)
    #     nullable = na_rep is None or self.parent.dtype.nullable

    #     res = StringColumn(Int64(nullable))
    #     if na_rep is None:
    #         #Todo should chcek fro same length
    #         for i,j in zip(data,others):
    #             if i is None or j is None:
    #                 res.append(None)
    #             else:
    #                 res.append(i+sep+j)
    #     else:
    #         for i,j in zip(data._iter_fill_na(na_rep),others._iter_fill_na(na_rep)):
    #             res.append(i+sep+j)
    #     return res

    def slice(
        self, start: int = None, stop: int = None, step: int = None
    ) -> StringColumn:
        """Slice substrings from each element in the data or Index."""
        data = self._parent
        res = StringColumn(String(data.dtype.nullable))
        for s in data:
            if s is None:
                res.append(None)
            else:
                res.append(s[start:stop:step])
        return res

    # CUDF only
    # def slice_from(
    #     self, starts: NumericalColumn, stops: NumericalColumn):
    #     """Slice substring from each element using start/stop positions for each string."""
    #     # stop = -1 is to the length of the string
    #     data = self._parent
    #     res = StringColumn(String(data.dtype.nullable))
    #     for i,start,stop in zip(data, starts, stops):
    #         if i is None or start is None or stop is None:
    #             res.append(None)
    #         else:
    #             if stop ==-1:
    #                 res.append(i[stop:])
    #             else:
    #                 res.append(i[stop:stop])
    #     return res

    # CUDF only
    # def slice_replace(
    #     self, start: int = None, stop: int = None, repl: str = None
    # ):
    #     """Replace the specified section of each string with a new string."""
    #     pass

    def split(self, sep=None, maxsplit=-1, expand=False):
        """Split strings around given separator/delimiter."""
        # if expand = True then return struct columns (labeled"0", "1")
        def fun(i): return i.split(sep, maxsplit)
        if not expand:
            return self._to_1_list(sep, maxsplit, direction='left')
        else:
            return self._to_n_lists(sep, maxsplit, direction='left')

    def _to_1_list(self, sep, maxsplit, direction):
        # cyclic import
        from .list_column import ListColumn

        assert direction in {'left', 'right'}

        me = self._parent
        fun = None
        if direction == "left":
            def fun(i): return i.split(sep, maxsplit)
        elif direction == 'right':
            def fun(i): return i.rsplit(sep, maxsplit)
        res = ListColumn(List_(me.dtype))
        for i in me:
            if i is None:
                res.append(None)
            else:
                res.append(fun(i))
        return res

    def _to_n_lists(self, sep, maxsplit, direction):
        # cyclic import
        from .dataframe import DataFrame

        assert direction in {'left', 'right'}
        assert maxsplit >= 0

        me = self._parent
        res = DataFrame(
            Struct([Field(str(i), String(nullable=True)) for i in range(maxsplit+1)]))
        for i in me:
            if i is None:
                res.append(tuple([None]*maxsplit))
            else:
                if direction == 'left':
                    ws = i.split(sep, maxsplit)
                    ws = ws + ([None] * (maxsplit+1-len(ws)))
                elif direction == 'right':
                    ws = i.rsplit(sep, maxsplit)
                    ws = ([None] * (maxsplit+1-len(ws)))+ws
                else:
                    raise AssertionError(
                        "direction must be in {'left', 'right'}")
                res.append(tuple(ws))
        return res

    def isinteger(self):
        """Check whether string forms a positive/negative integer"""
        def _isinteger(s):
            # could use regular expression instead (but would miss representation limits)
            try:
                _ = int(s)
                return True
            except ValueError:
                return False
        return self._map_boolean(_isinteger)

    def isfloat(self):
        """Check whether string forms a positive/negative floating point number"""
        def _isfloat(s):
            # could use regular expression instead (but would miss representation limits)
            try:
                _ = float(s)
                return True
            except ValueError:
                return False
        return self._map_boolean(_isfloat)

    def isalnum(self): return self._map_boolean(str.isalnum)
    def isalpha(self): return self._map_boolean(str.isalpha)
    def isascii(self): return self._map_boolean(str.isascii)
    def isdecimal(self): return self._map_boolean(str.isdecimal)
    def isdigit(self): return self._map_boolean(str.isdigit)
    def isidentifier(self): return self._map_boolean(str.isidentifier)
    def islower(self): return self._map_boolean(str.islower)
    def isnumeric(self): return self._map_boolean(str.isnumeric)
    def isprintable(self): return self._map_boolean(str.isprintable)
    def isspace(self): return self._map_boolean(str.isspace)
    def istitle(self): return self._map_boolean(str.istitle)
    def isupper(self): return self._map_boolean(str.isupper)

    def _map_boolean(self, pred):
        return self._map_boolean_na(pred, None)

    def _map_boolean_na(self, pred, na: Literal[True, False, None]):
        me = self._parent
        res = BooleanColumn(Boolean(me.dtype.nullable))
        for i in me:
            if i is None:
                res.append(na)
            else:
                res.append(pred(i))
        return res

    def _map_string(self, fun):
        me = self._parent
        res = StringColumn(me.dtype)
        for i in me:
            if i is None:
                res.append(None)
            else:
                res.append(fun(i))
        return res

    def _map_int64(self, fun):
        me = self._parent
        res = NumericalColumn(Int64(me.dtype.nullable))
        for i in me:
            if i is None:
                res.append(None)
            else:
                res.append(fun(i))
        return res

    def capitalize(self):
        """Convert strings in the data/Index to be capitalized"""
        return self._map_string(str.capitalize)

    def swapcase(self) -> StringColumn:
        """Change each lowercase character to uppercase and vice versa."""
        return self._map_string(str.swapcase)

    def title(self) -> StringColumn:
        return self._map_string(str.title)

    def lower(self) -> StringColumn:
        return self._map_string(str.lower)

    def upper(self) -> StringColumn:
        return self._map_string(str.upper)

    def casefold(self) -> StringColumn:
        return self._map_string(str.casefold)

    def repeat(self, repeats):
        """
        Duplicate each string in the data or Index.
        """
        pass

    def pad(self, width, side="left", fillchar=" "):
        fun = None
        if side == "left":
            def fun(i): return i.ljust(width, fillchar)
        if side == "right":
            def fun(i): return i.rjust(width, fillchar)
        if side == "center":
            def fun(i): return i.center(width, fillchar)
        return self._map_string(fun)

    def ljust(self, width, fillchar=" "):
        def fun(i): return i.ljust(width, fillchar)
        return self._map_string(fun)

    def rjust(self, width, fillchar=" "):
        def fun(i): return i.rjust(width, fillchar)
        return self._map_string(fun)

    def center(self, width, fillchar=" "):
        def fun(i): return i.center(width, fillchar)
        return self._map_string(fun)

    def zfill(self, width):
        def fun(i): return i.zfill(width)
        return self._map_string(fun)

    def translate(self, table):
        def fun(i): return i.translate(table)
        return self._map_string(fun)

    def count(self, pat, flags=0):
        """Count occurrences of pattern in each string"""
        assert flags == 0
        # TODOL for now just count of FIXED strings, i..e no RE
        def fun(i): return i.count(pat)
        return self._map_int64(fun)

    def startswith(self, pat, na: Literal[True, False, None] = None):
        """Test if the beginning of each string element matches a pattern."""
        def pred(i): return i.startswith(pat)
        return self._map_boolean_na(pred, na)

    def endswith(self, pat, na: Literal[True, False, None] = None):
        """Test if the end of each string element matches a pattern."""
        def pred(i): return i.endswith(pat)
        return self._map_boolean_na(pred, na)

    def find(self, sub, start=0, end=None):
        def fun(i): return i.find(sub, start, end)
        return self._map_int64(fun)

    def rfind(self, sub, start=0, end=None):
        def fun(i): return i.rfind(sub, start, end)
        return self._map_int64(fun)

    def index(self, sub, start=0, end=None):
        def fun(i):
            try:
                return i.index(sub, start, end)
            except ValueError:
                return -1
        return self._map_int64(fun)

    def rindex(self, sub, start=0, end=None):
        def fun(i):
            try:
                return i.rindex(sub, start, end)
            except ValueError:
                return -1
        return self._map_int64(fun)
