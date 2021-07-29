import torcharrow.dtypes as dt
from torcharrow.column_factory import Device

from .icolumn import IColumn


class INumericalColumn(IColumn):
    """Abstract Numerical Column"""

    # private
    def __init__(self, scope, device, dtype):  # , data, mask):
        assert dt.is_boolean_or_numerical(dtype)
        super().__init__(scope, device, dtype)

    # Note all numerical column implementations inherit from INumericalColumn

    def to(self, device: Device):
        from .numpy_rt import NumericalColumnStd
        from .velox_rt import NumericalColumnCpu

        if self.device == device:
            return self
        elif isinstance(self, NumericalColumnStd):
            return self.scope._FullColumn(
                self._data, self.dtype, device=device, mask=self._mask
            )
        elif isinstance(self, NumericalColumnCpu):
            return self.scope._FullColumn(
                self._data, self.dtype, device=device, mask=self._mask
            )
        else:
            raise AssertionError("unexpected case")
