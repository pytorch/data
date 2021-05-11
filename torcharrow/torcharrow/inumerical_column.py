from .icolumn import IColumn

from torcharrow.column_factory import Device
import torcharrow.dtypes as dt


class INumericalColumn(IColumn):
    """Abstract Numerical Column"""

    # private
    def __init__(self, scope, to, dtype):  # , data, mask):
        assert dt.is_boolean_or_numerical(dtype)
        super().__init__(scope, to, dtype)

    # Note all numerical column implementations inherit from INumericalColumn

    def move_to(self, to: Device):
        from .numpy_rt import NumericalColumnStd
        from .velox_rt import NumericalColumnCpu

        if self.to == to:
            return self
        elif isinstance(self, NumericalColumnStd):
            return self.scope._FullColumn(
                self._data, self.dtype, to=to, mask=self._mask
            )
        elif isinstance(self, NumericalColumnCpu):
            return self.scope._FullColumn(
                self._data, self.dtype, to=to, mask=self._mask
            )
        else:
            raise AssertionError("unexpected case")
