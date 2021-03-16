import array as ar
import copy
from dataclasses import dataclass

from .column import AbstractColumn, _set_column_constructor
from .dtypes import NL, is_boolean, is_numerical
from .tabulate import tabulate

# ------------------------------------------------------------------------------
# NumericalColumn


class NumericalColumn(AbstractColumn):

    def __init__(self, dtype, kwargs=None):
        assert is_numerical(dtype) or is_boolean(dtype)
        super().__init__(dtype)
        self._data = ar.array(dtype.arraycode)

    # implementing abstract methods ----------------------------------------------
    def _raw_lengths(self):
        return [len(self._data)]

    @property
    def ismutable(self):
        """Can this column/frame be extended without side effecting """
        return self._raw_lengths()[0] == self._offset + self._length

    def memory_usage(self, deep=False):
        """Return the memory usage of the column/frame (if deep then include buffer sizes)."""
        vsize = self._validity.itemsize
        dsize = self._data.itemsize
        if not deep:
            return self._length * vsize + self._length * dsize
        else:
            return len(self._validity)*vsize + len(self._data)*dsize

    def _copy(self, deep, offset, length):
        if deep:
            res = NumericalColumn(self.dtype)
            res._length = length
            res._data = self._data[offset: offset+length]
            res._validity = self._validity[offset: offset+length]
            res._null_count = sum(res._validity)
            return res
        else:
            return copy.copy(self)

    def append(self, x):
        """Append value to the end of the column/frame"""
        if x is None:
            if self._dtype.nullable:
                self._data.append(self._dtype.default)
                self._validity.append(False)
                self._null_count += 1
            else:
                self._data.append(None)  # throws TypeError (for None)
        else:
            # print('case NOT None', x)
            # throws TypeError (for any non expected type)
            self._data.append(x)
            self._validity.append(True)
        self._length += 1

    def get(self, i, fill_value):
        """Get item from column/frame for given integer index"""
        if self._null_count == 0:
            return self._data[self._offset+i]
        elif not self._validity[self._offset+i]:
            return fill_value
        else:
            return self._data[self._offset+i]

    def __iter__(self):
        """Return the iterator object itself."""
        if self._null_count == 0:
            for i in range(self._length):
                yield self._data[self._offset+i]
        else:
            for i in range(self._length):
                if self._validity[self._offset+i]:
                    yield self._data[self._offset+i]
                else:
                    yield None

    # printing ----------------------------------------------------------------
    def __str__(self):
        return f"Column([{', '.join(str(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate([[l if l is not None else 'None']
                       for l in self], tablefmt='plain', showindex=True)
        typ = f"dtype: {self._dtype}, length: {self._length}, null_count: {self._null_count}"
        return tab+NL+typ

    def show_details(self):
        return _NumericalRepr(self)


@dataclass
class _NumericalRepr:
    parent: NumericalColumn

    def __repr__(self):
        me = self.parent
        tab = tabulate([[l if l is not None else 'None', v] for (
            l, v) in zip(me._data, me._validity)], ['data', 'validity'])
        typ = f"dtype: {me._dtype}, count: {me._length}, null_count: {me._null_count}, offset: {me._offset}"
        return tab+NL+typ
# ------------------------------------------------------------------------------
# BooleanColumn


class BooleanColumn(NumericalColumn):

    def __init__(self, dtype, kwargs=None):
        assert is_boolean(dtype)
        super().__init__(dtype)
        self._data = ar.array('b')

    # implementing abstract methods ----------------------------------------------
    def get(self, i, fill_value):
        """Get item from column/frame for given integer index"""
        if self._null_count == 0:
            return bool(self._data[self._offset+i])
        elif not self._validity[self._offset+i]:
            return fill_value
        else:
            return bool(self._data[self._offset+i])

    def __iter__(self):
        """Return the iterator object itself."""
        if self._null_count == 0:
            # print('case _null_count==0')
            for i in range(self._length):
                yield bool(self._data[self._offset+i])
        else:
            # print('case _null_count>0')
            for i in range(self._length):
                if self._validity[self._offset+i]:
                    yield bool(self._data[self._offset+i])
                else:
                    yield None

    # printing ----------------------------------------------------------------
    def __str__(self):
        return f"Column([{', '.join(str(bool(i)) for i in self._data)}])"

    def __repr__(self):
        tab = tabulate([[str(bool(l)) if l is not None else 'None']
                       for l in self], tablefmt='plain', showindex=True)
        typ = f"dtype: {self._dtype}, count: {self._length}, null_count: {self._null_count}"
        return tab+NL+typ

    def show_details(self):
        return _BooleanRepr(self)


@dataclass
class _BooleanRepr:
    parent: NumericalColumn

    def __repr__(self):
        me = self.parent
        tab = tabulate([[str(bool(l)) if l is not None else 'None', v] for (
            l, v) in zip(me._data, me._validity)], ['data', 'validity'])
        typ = f"dtype: {me._dtype}, count: {me._length}, null_count: {me._null_count}, offset: {me._offset}"
        return tab+NL+typ


# ------------------------------------------------------------------------------
# registering the factory
_set_column_constructor(is_numerical, NumericalColumn)
_set_column_constructor(is_boolean, BooleanColumn)
