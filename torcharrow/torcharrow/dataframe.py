import math
import operator
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import stat
from typing import (
    Callable,
    Sequence,
    Type,
    Union,
    Optional,
    List,
    Dict,
    ClassVar,
    Callable,
    Any,
)

import _torcharrow as ta

# high level stuff -- all types are described here
from .dtypes import (
    DType,
    MetaData,
    derive_operator,
    String,
    string,
    boolean,
    Int64,
    Boolean,
    int64,
    List_,
    Struct,
    is_numerical,
    is_string,
    is_boolean,
    is_primitive,
    is_map,
    is_struct,
    is_list,
    infer_dtype,
    derive_dtype,
    derive_operator,
    Field,
    Schema,
    Map,
)


def get_velox_type(dtype: DType):
    if dtype == int64:
        return ta.INTEGER()
    elif dtype == string:
        return ta.VARCHAR()
    elif dtype == boolean:
        return ta.BOOLEAN()
    elif isinstance(dtype, List_):
        return ta.ARRAY(get_velox_type(dtype.item_dtype))
    elif isinstance(dtype, Map):
        return ta.MAP(get_velox_type(dtype.key_dtype), get_velox_type(dtype.item_dtype))
    else:
        raise NotImplementedError()


@dataclass
class _Column(ABC):
    _dtype: DType
    _data: ta.BaseColumn

    def __len__(self):
        return len(self._data)

    @property
    def _validity(self):
        return self._data.get_validity()

    def __iter__(self):
        validity = self._validity
        for i in range(len(self)):
            if not validity[i]:
                yield None
            else:
                yield self[i]

    @property
    def _null_count(self):
        return sum(1 for non_none in self._validity if not non_none)


@dataclass
class _SimpleColumn(_Column, ABC):
    def append(self, value):
        if value is None and self._dtype.nullable:
            self._data.appendNull()
        else:
            self._data.append(value)

    def __getitem__(self, i):
        if isinstance(i, int):
            i = i % len(self)
            if self._validity[i]:
                return self._data[i]
            else:
                return None
        elif isinstance(i, slice):
            # TODO generalize this...
            assert i.start is not None and i.stop is not None
            # TODO fix this: e.g. by creating a new columns instead
            # res = _create_column(self._dtype)
            # for j in range(i.start, i.stop):
            #     res.append( self.get(j, None))
            # return res
            # # but this requires  changes in tests, so skip for now!
            return [self[j] for j in range(i.start, i.stop)]
        elif isinstance(i, _BooleanColumn):
            res = _create_column(self._dtype)
            for x, m in zip(list(self), list(i)):
                if m:
                    res.append(x)
            return res
        else:
            raise NotImplementedError(str(i))

    def extend(self, iterable):
        if iterable is not None:
            for i in iterable:
                self.append(i)
        else:
            breakpoint

    def fillna(self, fill_value):
        assert fill_value is not None
        res = _create_column(self._dtype)
        for i in range(len(self)):
            if self._validity[i]:
                res.append(self[i])
            else:
                res.append(fill_value)
        return res

    # binary operators --------------------------------------------------------

    def _broadcast(self, operator, const, dtype):
        assert is_primitive(self._dtype) and is_primitive(dtype)
        res = _create_column(dtype)
        for i in range(len(self)):
            if self._validity[i]:
                res.append(operator(self._data[i], const))
            else:
                res.append(None)
        return res

    def _pointwise(self, operator, other, dtype):
        assert is_primitive(self._dtype) and is_primitive(dtype)
        res = _create_column(dtype)
        for i in range(len(self)):
            if self._validity[i] and other._validity[i]:
                res.append(operator(self._data[i], other._data[i]))
            else:
                res.append(None)
        return res

    def _binary_operator(self, operator, other):
        if isinstance(other, (int, float, list, set, type(None))):
            return self._broadcast(
                derive_operator(operator), other, derive_dtype(self._dtype, operator)
            )
        else:
            return self._pointwise(
                derive_operator(operator), other, derive_dtype(self._dtype, operator)
            )

    def __add__(self, other):
        return self._binary_operator("add", other)

    def __sub__(self, other):
        return self._binary_operator("sub", other)

    def __mul__(self, other):
        return self._binary_operator("mul", other)

    def __eq__(self, other):
        return self._binary_operator("eq", other)

    def __ne__(self, other):
        return self._binary_operator("ne", other)

    def __or__(self, other):
        return self._binary_operator("or", other)

    def __and__(self, other):
        return self._binary_operator("and", other)

    def __floordiv__(self, other):
        return self._binary_operator("floordiv", other)

    def __truediv__(self, other):
        return self._binary_operator("truediv", other)

    def __mod__(self, other):
        return self._binary_operator("mod", other)

    def __pow__(self, other):
        return self._binary_operator("pow", other)

    def __lt__(self, other):
        return self._binary_operator("lt", other)

    def __gt__(self, other):
        return self._binary_operator("gt", other)

    def __le__(self, other):
        return self._binary_operator("le", other)

    def __ge__(self, other):
        return self._binary_operator("ge", other)

    def isin(self, collection):
        return self._binary_operator("in", collection)


class _BooleanColumn(_SimpleColumn):
    def __init__(self, dtype: DType, velox_column: Optional[ta.BaseColumn] = None):
        super().__init__(dtype, velox_column or ta.Column(ta.BOOLEAN()))


class _NumericalColumn(_SimpleColumn):
    def __init__(self, dtype: DType, velox_column: Optional[ta.BaseColumn] = None):
        super().__init__(dtype, velox_column or ta.Column(ta.INTEGER()))

    # descriptive statistics --------------------------------------------------
    # special operations on Numerical Columns

    def sum(self):
        if self._null_count == 0:
            return sum(self._data)
        else:
            return sum(self._data[i] for i in range(len(self)) if self._validity[i])

    def mean(self):
        if self._null_count == 0:
            return statistics.mean(self._data)
        else:
            return statistics.mean(
                self._data[i] for i in range(len(self)) if self._validity[i]
            )


class _StringColumn(_SimpleColumn):
    def __init__(self, dtype: DType, velox_column: Optional[ta.BaseColumn] = None):
        super().__init__(dtype, velox_column or ta.Column(ta.VARCHAR()))


# -----------------------------------------------------------------------------
# List


@dataclass
class Offsets:
    _data: ta.ArrayColumn

    def __getitem__(self, index):
        if index < len(self._data):
            return self._data.offset_at(index)
        elif index == len(self._data):
            return self._data.get_elements_size()
        else:
            raise ValueError(f"Index {index} is out of range")

    def __len__(self):
        return len(self._data) + 1


class _ListColumn(_Column):
    def __init__(self, dtype, velox_column: Optional[ta.BaseColumn] = None):
        super().__init__(dtype, velox_column or ta.Column(get_velox_type(dtype)))
        self._offsets = Offsets(self._data)

    def append(self, values):
        if values is None:
            self._data.appendNull()
        else:
            new_element_column = _create_column(self._dtype.item_dtype)
            new_element_column.extend(values)
            self._data.append(new_element_column._data)

    def __getitem__(self, i) -> List[Any]:
        if isinstance(i, int):
            i = i % len(self)
            if self._validity[i]:
                return list(_create_column(self._dtype.item_dtype, self._data[i]))
            else:
                return None
        else:
            raise NotImplementedError()

    def extend(self, iterable):
        if iterable is not None:
            for i in iterable:
                self.append(i)
        else:
            breakpoint()

    # # list_ops-----------------------------------------------------------------

    # def count(self, x):
    #     "total number of occurrences of x in s"
    #     res = _NumericalColumn(Int64(self._dtype.nullable))
    #     for i in self.iter():
    #         res.append(i._count(x))
    #     return res


# ops on list  --------------------------------------------------------------
#  'count',
#  'extend',
#  'index',
#  'insert',
#  'pop',
#  'remove',
#  'reverse',


# -----------------------------------------------------------------------------
# Map


class _MapColumn(_Column):
    def __init__(self, dtype, velox_column: Optional[ta.BaseColumn] = None):
        super().__init__(dtype, velox_column or ta.Column(get_velox_type(dtype)))

    def append(self, value: Dict[Any, Any]):
        if value is None:
            raise NotImplementedError()
        else:
            new_key = _create_column(self._dtype.key_dtype)
            new_value = _create_column(self._dtype.item_dtype)
            for k, v in value.items():
                new_key.append(k)
                new_value.append(v)
            self._data.append(new_key._data, new_value._data)

    def __getitem__(self, i) -> Dict[Any, Any]:
        if isinstance(i, int):
            i = i % len(self)
            if self._validity[i]:
                items = self._data[i]
                return {
                    k: v
                    for k, v in zip(
                        _create_column(self._dtype.key_dtype, items.keys()),
                        _create_column(self._dtype.item_dtype, items.values()),
                    )
                }
            else:
                return None
        else:
            raise NotImplementedError()


# ops on maps --------------------------------------------------------------
#  'get',
#  'items',
#  'keys',
#  'pop',
#  'popitem',
#  'setdefault',
#  'update',
#  'values'

# -----------------------------------------------------------------------------
# Column and Dataframe factories.
# -- note that Dataframe is in qutessence an alias for a _StructColumn

# public factory API


def DataFrame(
    initializer: Union[Dict, DType, None] = None, dtype: Optional[DType] = None
) -> _Column:
    if initializer is None and dtype is None:
        return _StructColumn(Schema([]), {})

    if isinstance(initializer, DType):
        assert dtype is None
        dtype = initializer
        initializer = None

    if dtype is not None:
        col = _create_column(dtype)
        if initializer is not None:
            for i in initializer:
                col.append(i)
        return col
    else:
        if isinstance(initializer, dict):
            cols = {}
            fields = []
            for k, vs in initializer.items():
                cols[k] = Column(vs)
                fields.append(Field(k, cols[k]._dtype))
            return _StructColumn(Schema(fields), cols)
        else:
            raise ValueError("cannot infer type of initializer")


def Column(
    initializer: Union[Dict, List, DType, None] = None, dtype: Optional[DType] = None
) -> _Column:
    if isinstance(initializer, DType):
        assert dtype is None
        dtype = initializer
        initializer = None

    if dtype is not None:
        col = _create_column(dtype)
        if initializer is not None:
            for i in initializer:
                col.append(i)
        return col
    elif isinstance(initializer, dict):
        cols = {}
        fields = []
        for k, vs in initializer.items():
            cols[k] = Column(vs)
            fields.append(Field(k, cols[k]._dtype))
        return _StructColumn(Struct(fields), cols)
    elif isinstance(initializer, list):
        dtype = infer_dtype(initializer[0:5])
        if dtype is None:
            raise ValueError("cannot infer type of initializer")
        col = _create_column(dtype)
        for i in initializer:
            col.append(i)
        return col
    else:
        raise ValueError("cannot infer type of initializer")


# private factory method
def _create_column(
    dtype: Optional[DType] = None, velox_column: Optional[ta.BaseColumn] = None
):
    # if dtype is None:
    #     return _StructColumn(Struct([]))
    if is_numerical(dtype):
        return _NumericalColumn(dtype, velox_column)
    elif is_boolean(dtype):
        return _BooleanColumn(dtype, velox_column)
    elif is_string(dtype):
        return _StringColumn(dtype, velox_column)
    elif is_list(dtype):
        return _ListColumn(dtype, velox_column)
    elif is_map(dtype):
        return _MapColumn(dtype, velox_column)
    else:
        raise NotImplementedError(f"{dtype}")
    # if is_struct(dtype):
    #     return _StructColumn(
    #         dtype, {f.name: _create_column(f.dtype) for f in dtype.fields}
    #     )
    # raise AssertionError(f"unexpected case: {dtype}")
