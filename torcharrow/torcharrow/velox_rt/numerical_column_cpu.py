import array as ar
from typing import Dict, List, Literal, Optional, Union, cast

import numpy as np

import torcharrow.dtypes as dt
from torcharrow.icolumn import IColumn
from torcharrow.inumerical_column import INumericalColumn
from torcharrow.scope import ColumnFactory
from torcharrow.trace import trace

# ------------------------------------------------------------------------------


class NumericalColumnCpu(INumericalColumn):
    """A Numerical Column"""

    # NumericalColumnCpu is currently exactly the same code as
    # NumericalColumnStd
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
        self._data = data  # Union[ar.array.np.ndarray]
        self._mask = mask  # Union[ar.array.np.ndarray]

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
        self._mask.append(True)
        self._data.append(self.dtype.default)

    def _append_value(self, value):
        self._mask.append(False)
        if isinstance(value, np.bool_):
            # TODO Get rid of case. Currently required due to Numpy 's
            # DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
            self._data.append(bool(value))
        else:
            self._data.append(value)

    def _append_data(self, value):
        self._data.append(value)

    def _finalize(self):
        self._data = np.array(
            self._data, dtype=dt.np_typeof_dtype(self.dtype), copy=False
        )
        if isinstance(self._mask, (bool, np.bool8)):
            self._mask = np.full((len(self._data),), self._mask, dtype=np.bool8)
        elif isinstance(self._mask, ar.array):
            self._mask = np.array(self._mask, dtype=np.bool8, copy=False)
        else:
            assert isinstance(self._mask, np.ndarray)
        return self

    def _valid_mask(self, ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def __len__(self):
        return len(self._data)

    def null_count(self):
        """Return number of null items"""
        return sum(self._mask) if self.isnullable else 0

    @trace
    def copy(self):
        return self.scope._FullColumn(self._data.copy(), self.mask.copy())

    def getdata(self, i):
        return self._data[i]

    def getmask(self, i):
        return self._mask[i]


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
