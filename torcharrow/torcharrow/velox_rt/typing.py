import _torcharrow as velox
from torcharrow.dtypes import (
    get_underlying_dtype,
    DType,
    int8,
    int16,
    int32,
    int64,
    float64,
    string,
    boolean,
    List as List_,
    Map,
    Struct,
)


# ------------------------------------------------------------------------------


def get_velox_type(dtype: DType):
    underlying_dtype = get_underlying_dtype(dtype)
    if underlying_dtype == int64:
        return velox.BIGINT()
    elif underlying_dtype == int32:
        return velox.INTEGER()
    elif underlying_dtype == int16:
        return velox.SMALLINT()
    elif underlying_dtype == int8:
        return velox.TINYINT()
    elif underlying_dtype == float64:
        return velox.DOUBLE()
    elif underlying_dtype == string:
        return velox.VARCHAR()
    elif underlying_dtype == boolean:
        return velox.BOOLEAN()
    elif isinstance(underlying_dtype, List_):
        return velox.ARRAY(get_velox_type(underlying_dtype.item_dtype))
    elif isinstance(underlying_dtype, Map):
        return velox.MAP(
            get_velox_type(underlying_dtype.key_dtype),
            get_velox_type(underlying_dtype.item_dtype),
        )
    elif isinstance(underlying_dtype, Struct):
        return velox.ROW(
            [f.name for f in underlying_dtype.fields],
            [get_velox_type(f.dtype) for f in underlying_dtype.fields],
        )
    else:
        raise NotImplementedError(str(underlying_dtype))
