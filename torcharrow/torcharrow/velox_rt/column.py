import _torcharrow as velox
from torcharrow.column_factory import Device
from torcharrow.dtypes import DType
from torcharrow.icolumn import IColumn
from torcharrow.scope import Scope


class ColumnFromVelox:
    _data: velox.BaseColumn
    _finialized: bool

    @staticmethod
    def from_velox(
        scope: Scope, device: Device, dtype: DType, data: velox.BaseColumn, finialized: bool
    ) -> IColumn:
        col = scope.Column(dtype=dtype, device=device)
        col._data = data
        col._finialized = finialized
        return col

    # Velox column returned from generic dispatch always assumes returned column is nullable
    # This help method allows to alter it based on context (e.g. methods in IStringMethods can have better inference)
    def with_null(self, nullable: bool):
        return self.from_velox(
            self.scope,
            self.device,
            self.dtype.with_null(nullable),
            self._data,
            True
        )
