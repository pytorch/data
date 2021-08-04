from .numerical_column_cpu import *
from .string_column_cpu import *
from .list_column_cpu import *
from .map_column_cpu import *
from .dataframe_cpu import *
import _torcharrow
import torcharrow

_torcharrow.BaseColumn.dtype = lambda self: torcharrow.dtypes.dtype_of_velox_type(
    self.type()
)
