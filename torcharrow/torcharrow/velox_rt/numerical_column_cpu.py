import array as ar
import math
import operator
import statistics
from typing import Dict, List, Literal, Optional, Union, cast

import _torcharrow as velox
import numpy as np
import torcharrow.dtypes as dt
from torcharrow.expression import expression
from torcharrow.icolumn import IColumn
from torcharrow.inumerical_column import INumericalColumn
from torcharrow.scope import ColumnFactory
from torcharrow.trace import trace

from .column import ColumnFromVelox
from .typing import get_velox_type

# ------------------------------------------------------------------------------


class NumericalColumnCpu(INumericalColumn, ColumnFromVelox):
    """A Numerical Column"""

    # NumericalColumnCpu is currently exactly the same code as
    # NumericalColumnCpu
    #
    # However it uses the factory key 'cpu',e.g. see
    #
    # ColumnFactory.register(
    #       (dtype.typecode+"_empty", 'cpu'), NumericalColumnCpu._empty)
    # ...

    # Velox can implement whatever it wants to implement,
    # the only contract it has to obey is that
    # - the signatures for all public APIs must stay the same
    # - the signature of the internal builders must stay the same, e.g
    # _full, _empty, _append_null, _append_value, _append_data, _finalize

    # private
    def __init__(self, scope, to, dtype, data, mask):
        assert dt.is_boolean_or_numerical(dtype)
        super().__init__(scope, to, dtype)
        self._data = velox.Column(get_velox_type(dtype))
        for m, d in zip(mask.tolist(), data.tolist()):
            if m:
                self._data.append_null()
            else:
                self._data.append(d)
        self._finialized = False

    @staticmethod
    def _full(scope, to, data, dtype=None, mask=None):
        assert isinstance(data, np.ndarray) and data.ndim == 1
        if dtype is None:
            dtype = dt.typeof_np_ndarray(data.dtype)
        else:
            if dtype != dt.typeof_np_dtype(data.dtype):
                # TODO fix nullability
                # raise TypeError(f'type of data {data.dtype} and given type {dtype} must be the same')
                pass
        if not dt.is_boolean_or_numerical(dtype):
            raise TypeError(f"construction of columns of type {dtype} not supported")
        if mask is None:
            mask = NumericalColumnCpu._valid_mask(len(data))
        elif len(data) != len(mask):
            raise ValueError(
                f"data length {len(data)} must be the same as mask length {len(mask)}"
            )
        # TODO check that all non-masked items are legal numbers (i.e not nan)
        return NumericalColumnCpu(scope, to, dtype, data, mask)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(scope, to, dtype, mask=None):
        _mask = mask if mask is not None else ar.array("b")
        return NumericalColumnCpu(scope, to, dtype, ar.array(dtype.arraycode), _mask)

    def _append_null(self):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        if isinstance(value, np.bool_):
            # TODO Get rid of case. Currently required due to Numpy 's
            # DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
            self._data.append(bool(value))
        else:
            self._data.append(value)

    def _append_data(self, value):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        self._data.append(value)

    def _finalize(self):
        self._finialized = True
        return self

    def _valid_mask(self, ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def __len__(self):
        return len(self._data)

    def null_count(self):
        """Return number of null items"""
        return self._data.get_null_count()

    @trace
    def copy(self):
        return self.scope._FullColumn(self._data.copy(), self.mask.copy())

    def getdata(self, i):
        if i < 0:
            i += len(self._data)
        if self._data.is_null_at(i):
            return self.dtype.default
        else:
            return self._data[i]

    def getmask(self, i):
        if i < 0:
            i += len(self._data)
        return self._data.is_null_at(i)

    # if-then-else ---------------------------------------------------------------

    def ite(self, then_, else_):
        """Vectorized if-then-else"""
        if not dt.is_boolean(self.dtype):
            raise TypeError("condition must be a boolean vector")
        if not isinstance(then_, IColumn):
            then_ = self._Column(then_)
        if not isinstance(else_, IColumn):
            else_ = self._Column(else_)
        lub = dt.common_dtype(then_.dtype, else_.dtype)

        if lub is None or dt.is_void(lub):
            raise TypeError(
                "then and else branches must have compatible types, got {then_.dtype} and {else_.dtype}, respectively"
            )
        if isinstance(then_, NumericalColumnCpu) and isinstance(
            else_, NumericalColumnCpu
        ):
            col = velox.Column(get_velox_type(lub))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(then_.getdata(i) if self.getdata(i) else else_.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, lub, col, True)

        else:
            # refer back to default handling...
            return super.ite(self, then_, else_)

    # sorting, top-k, unique---------------------------------------------------

    @trace
    @expression
    def sort(
        self,
        columns: Optional[List[str]] = None,
        ascending=True,
        na_position: Literal["last", "first"] = "last",
    ):
        """Sort a column/a dataframe in ascending or descending order"""
        if columns is not None:
            raise TypeError("sort on numerical column can't have 'columns' parameter")
        res = []
        none_count = 0
        for i in range(len(self)):
            if self.getmask(i):
                none_count += 1
            else:
                res.append(self.getdata(i))
        res.sort(reverse=not ascending)

        col = velox.Column(get_velox_type(self.dtype))
        if na_position == "first":
            for i in range(none_count):
                col.append_null()
        for value in res:
            col.append(value)
        if na_position == "last":
            for i in range(none_count):
                col.append_null()

        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def nlargest(
        self,
        n=5,
        columns: Optional[List[str]] = None,
        keep: Literal["last", "first"] = "first",
    ):
        """Returns a new data of the *n* largest element."""
        if columns is not None:
            raise TypeError(
                "computing n-largest on numerical column can't have 'columns' parameter"
            )
        return self.sort(columns=None, ascending=False, na_position=keep).head(n)

    @trace
    @expression
    def nsmallest(self, n=5, columns: Optional[List[str]] = None, keep="first"):
        """Returns a new data of the *n* smallest element."""
        if columns is not None:
            raise TypeError(
                "computing n-smallest on numerical column can't have 'columns' parameter"
            )

        return self.sort(columns=None, ascending=True, na_position=keep).head(n)

    @trace
    @expression
    def nunique(self, dropna=True):
        """Returns the number of unique values of the column"""
        result = set()
        for i in range(len(self)):
            if self.getmask(i):
                if not dropna:
                    result.add(None)
            else:
                result.add(self.getdata(i))
        return len(result)


    # operators ---------------------------------------------------------------

    @trace
    @expression
    def __add__(self, other):
        """Vectorized a + b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) + other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) + other)
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)


    @trace
    @expression
    def __radd__(self, other):
        """Vectorized b + a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(other.getdata(i) + self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(other + self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def __sub__(self, other):
        """Vectorized a - b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) - other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) - other)
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def __rsub__(self, other):
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(other.getdata(i) - self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(other - self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def __mul__(self, other):
        """Vectorized a * b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) * other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) * other)
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
    @trace
    @expression
    def __rmul__(self, other):
        """Vectorized b * a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(other.getdata(i) * self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(other * self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def __floordiv__(self, other):
        """Vectorized a // b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(dt.float64))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) // other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.float64, col, True)
        else:
            col = velox.Column(get_velox_type(dt.float64))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) // other)
            return ColumnFromVelox.from_velox(self.scope, dt.float64, col, True)
    @trace
    @expression
    def __rfloordiv__(self, other):
        """Vectorized b // a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(other.getdata(i) // self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(other // self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def __truediv__(self, other):
        """Vectorized a / b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(dt.float64))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) / other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.float64, col, True)
        else:
            col = velox.Column(get_velox_type(dt.float64))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) / other)
            return ColumnFromVelox.from_velox(self.scope, dt.float64, col, True)
    @trace
    @expression
    def __rtruediv__(self, other):
        """Vectorized b / a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(dt.float64))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(other.getdata(i) / self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.float64, col, True)
        else:
            col = velox.Column(get_velox_type(dt.float64))
            for i in range(len(self)):
                if self.getmask(i) or self.getdata(i) == 0:
                    col.append_null()
                else:
                    col.append(other / self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.float64, col, True)

    @trace
    @expression
    def __mod__(self, other):
        """Vectorized a % b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(self._ma() % other._ma())
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) % other)
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def __rmod__(self, other):
        """Vectorized b % a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            return self._from_ma(other._ma() + self._ma())
        else:
            return self._from_ma(other + self._ma())

    @trace
    @expression
    def __pow__(self, other):
        """Vectorized a ** b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) ** other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) ** other)
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def __rpow__(self, other):
        """Vectorized b ** a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(self.dtype))
            assert len(self) == len(other)
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(other.getdata(i) ** self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(other ** self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)
    @trace
    @expression
    def __eq__(self, other):
        """Vectorized a == b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) == other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) == other)
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __ne__(self, other):
        """Vectorized a != b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            raise NotImplementedError()
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) != other)
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __lt__(self, other):
        """Vectorized a < b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            self.scope.check_is_same(other.scope)
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) < other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) < other)
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __gt__(self, other):
        """Vectorized a > b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            self.scope.check_is_same(other.scope)
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) > other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) > other)
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __le__(self, other):
        """Vectorized a < b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            raise NotImplementedError()
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) <= other)
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __ge__(self, other):
        """Vectorized a <= b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) >= other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) >= other)
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __or__(self, other):
        """Vectorized boolean or: a | b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) | other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) | other)
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __ror__(self, other):
        """Vectorized boolean reverse or: b | a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            raise NotImplementedError()
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(other | self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __and__(self, other):
        """Vectorized boolean and: a & b."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i) or other.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) & other.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(self.getdata(i) & other)
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __rand__(self, other):
        """Vectorized boolean reverse and: b & a."""
        if isinstance(other, INumericalColumn):
            self.scope.check_is_same(other.scope)
        if isinstance(other, NumericalColumnCpu):
            raise NotImplementedError()
        else:
            col = velox.Column(get_velox_type(dt.boolean))
            for i in range(len(self)):
                if self.getmask(i):
                    col.append_null()
                else:
                    col.append(other & self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, dt.boolean, col, True)

    @trace
    @expression
    def __invert__(self):
        """Vectorized boolean not: ~ a."""
        col = velox.Column(get_velox_type(self.dtype))
        for i in range(len(self)):
            if self.getmask(i):
                col.append_null()
            else:
                col.append(not self.getdata(i))
        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def __neg__(self):
        """Vectorized: - a."""
        return ColumnFromVelox.from_velox(self.scope, self.dtype, self._data.neg(), True)

    @trace
    @expression
    def __pos__(self):
        """Vectorized: + a."""
        col = velox.Column(get_velox_type(self.dtype))
        for i in range(len(self)):
            if self.getmask(i):
                col.append_null()
            else:
                col.append(self.getdata(i))
        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def isin(self, values, invert=False):
        """Check whether list values are contained in data, or column/dataframe (row/column specific)."""
        # Todo decide on wether mask matters?
        if invert:
            raise NotImplementedError()
        col = velox.Column(get_velox_type(dt.boolean))
        for i in range(len(self)):
            if self.getmask(i):
                col.append(False)
            else:
                col.append(self.getdata(i) in values)
        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def abs(self):
        """Absolute value of each element of the series."""
        col = velox.Column(get_velox_type(self.dtype))
        for i in range(len(self)):
            if self.getmask(i):
                col.append_null()
            else:
                col.append(abs(self.getdata(i)))
        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def ceil(self):
        """Rounds each value upward to the smallest integral"""
        col = velox.Column(get_velox_type(self.dtype))
        for i in range(len(self)):
            if self.getmask(i):
                col.append_null()
            else:
                col.append(math.ceil(self.getdata(i)))
        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def floor(self):
        """Rounds each value downward to the largest integral value"""
        col = velox.Column(get_velox_type(self.dtype))
        for i in range(len(self)):
            if self.getmask(i):
                col.append_null()
            else:
                col.append(math.floor(self.getdata(i)))
        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def round(self, decimals=0):
        """Round each value in a data to the given number of decimals."""
        col = velox.Column(get_velox_type(self.dtype))
        for i in range(len(self)):
            if self.getmask(i):
                col.append_null()
            else:
                col.append(round(self.getdata(i), decimals))
        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    # data cleaning -----------------------------------------------------------

    @trace
    @expression
    def fillna(self, fill_value: Union[dt.ScalarTypes, Dict]):
        """Fill NA/NaN values using the specified method."""
        if not isinstance(fill_value, IColumn.scalar_types):
            raise TypeError(f"fillna with {type(fill_value)} is not supported")
        if not self.isnullable:
            return self
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    if isinstance(fill_value, Dict):
                        raise NotImplementedError()
                    else:
                        col.append(fill_value)
                else:
                    col.append(self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def dropna(self, how: Literal["any", "all"] = "any"):
        """Return a column with rows removed where a row has any or all nulls."""
        if not self.isnullable:
            return self
        else:
            col = velox.Column(get_velox_type(self.dtype))
            for i in range(len(self)):
                if self.getmask(i):
                    pass
                else:
                    col.append(self.getdata(i))
            return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    @trace
    @expression
    def drop_duplicates(
        self,
        subset: Optional[List[str]] = None,
    ):
        """Remove duplicate values from row/frame"""
        if subset is not None:
            raise TypeError(f"subset parameter for numerical columns not supported")
        seen = set()
        col = velox.Column(get_velox_type(self.dtype))
        for i in range(len(self)):
            if self.getmask(i):
                col.append_null()
            else:
                current = self.getdata(i)
                if current not in seen:
                    col.append(current)
                    seen.add(current)
        return ColumnFromVelox.from_velox(self.scope, self.dtype, col, True)

    # universal  ---------------------------------------------------------------

    @trace
    @expression
    def min(self, numeric_only=None, fill_value=None):
        """Return the minimum of the non-null values of the Column."""
        result = None
        for i in range(len(self)):
            if not self.getmask(i):
                value = self.getdata(i)
                if result is None or value < result:
                    result = value
        return result

    @trace
    @expression
    def max(self, fill_value=None):
        """Return the maximum of the non-null values of the column."""
        result = None
        for i in range(len(self)):
            if not self.getmask(i):
                value = self.getdata(i)
                if result is None or value > result:
                    result = value
        return result

    @trace
    @expression
    def all(self):
        """Return whether all non-null elements are True in Column"""
        for i in range(len(self)):
            if not self.getmask(i):
                value = self.getdata(i)
                if value == False:
                    return False
        return True

    @trace
    @expression
    def any(self, skipna=True, boolean_only=None):
        """Return whether any non-null element is True in Column"""
        for i in range(len(self)):
            if not self.getmask(i):
                value = self.getdata(i)
                if value == True:
                    return True
        return False

    @trace
    @expression
    def sum(self):
        # TODO Should be def sum(self, initial=None) but didn't get to work
        """Return sum of all non-null elements in Column (starting with initial)"""
        result = 0
        for i in range(len(self)):
            if not self.getmask(i):
                result += self.getdata(i)
        return result

    @trace
    @expression
    def prod(self):
        """Return produce of the values in the data"""
        result = 1
        for i in range(len(self)):
            if not self.getmask(i):
                result *= self.getdata(i)
        return result

    def _accumulate_column(self, func, *, skipna=True, initial=None):
        it = iter(self)
        # res = self.scope.Column(self.dtype)
        res = []
        total = initial
        rest_is_null = False
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                raise ValueError(f"cum[min/max] undefined for empty column.")
        if total is None:
            raise ValueError(f"cum[min/max] undefined for columns with row 0 as null.")

        res.append(total)
        for element in it:
            if rest_is_null:
                res.append(None)
                continue
            if element is None:
                if skipna:
                    res.append(None)
                else:
                    res.append(None)
                    rest_is_null = True
            else:
                total = func(total, element)
                res.append(total)
        return self.scope.Column(res, self.dtype)

    @trace
    @expression
    def cummin(self):
        """Return cumulative minimum of the data."""
        return self._accumulate_column(min, skipna=True, initial=None)

    @trace
    @expression
    def cummax(self):
        """Return cumulative maximum of the data."""
        return self._accumulate_column(max, skipna=True, initial=None)

    @trace
    @expression
    def cumsum(self):
        """Return cumulative sum of the data."""
        return self._accumulate_column(operator.add, skipna=True, initial=None)

    @trace
    @expression
    def cumprod(self):
        """Return cumulative product of the data."""
        return self._accumulate_column(operator.mul, skipna=True, initial=None)

    @trace
    @expression
    def mean(self):
        """Return the mean of the values in the series."""
        return statistics.mean(value for value in self if value is not None)

    @trace
    @expression
    def median(self):
        """Return the median of the values in the data."""
        return statistics.median(value for value in self if value is not None)

    # @ trace
    # @ expression
    # def mode(self):
    #     """Return the mode(s) of the data."""
    #     return np.ma.mode(self._ma())

    @trace
    @expression
    def Cpu(self, ddof=1):
        """Return the Cpudev(s) of the data."""
        # ignores nulls
        return np.ma.Cpu(self._ma(), ddof=ddof)

    @trace
    @expression
    def percentiles(self, q, interpolation="midpoint"):
        """Compute the q-th percentile of non-null data."""
        if len(self) == 0 or len(q) == 0:
            return []
        out = []
        s = sorted(self)
        for percent in q:
            k = (len(self) - 1) * (percent / 100)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                out.append(s[int(k)])
                continue
            d0 = s[int(f)] * (c - k)
            d1 = s[int(c)] * (k - f)
            out.append(d0 + d1)
        return out

    # unique and montonic  ----------------------------------------------------

    @trace
    @expression
    def is_unique(self):
        """Return boolean if data values are unique."""
        return self.nunique(dropna=False) == len(self)

    @trace
    @expression
    def is_monotonic_increasing(self):
        """Return boolean if values in the object are monotonic increasing"""
        first = True
        prev = None
        for i in range(len(self)):
            if not self.getmask(i):
                current = self.getdata(i)
                if not first:
                    if prev > current:
                        return False
                else:
                    first = False
                prev = current
        return True


    @trace
    @expression
    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonic decreasing"""
        first = True
        prev = None
        for i in range(len(self)):
            if not self.getmask(i):
                current = self.getdata(i)
                if not first:
                    if prev < current:
                        return False
                else:
                    first = False
                prev = current
        return True

    # interop ----------------------------------------------------------------

    @trace
    def to_pandas(self):
        """Convert self to pandas dataframe"""
        # TODO Add type translation
        # Skipping analyzing 'pandas': found module but no type hints or library stubs
        import pandas as pd  # type: ignore

        return pd.Series(self._ma())

    @trace
    def to_arrow(self):
        """Convert self to pandas dataframe"""
        # TODO Add type translation
        import pyarrow as pa  # type: ignore

        return pa.array(self._ma())


# ------------------------------------------------------------------------------
# registering all numeric and boolean types for the factory...
_primitive_types: List[dt.DType] = [
    dt.Int8(),
    dt.Int16(),
    dt.Int32(),
    dt.Int64(),
    dt.Float32(),
    dt.Float64(),
    dt.Boolean(),
]
for t in _primitive_types:
    ColumnFactory.register((t.typecode + "_empty", "cpu"), NumericalColumnCpu._empty)

# registering all numeric and boolean types for the factory...
for t in _primitive_types:
    ColumnFactory.register((t.typecode + "_full", "cpu"), NumericalColumnCpu._full)
