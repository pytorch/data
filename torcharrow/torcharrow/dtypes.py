import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Union, Tuple, Callable

# -----------------------------------------------------------------------------
# Aux

# Pretty printing constants; reused everywhere
OPEN = "{"
CLOSE = "}"
NL = "\n"

# Handy Type abbreviations; reused everywhere
ScalarTypes = Union[int, float, bool, str]
ScalarTypeValues = (int, float, bool, str)


# -----------------------------------------------------------------------------
# Schema and Field

MetaData = Dict[str, str]


@dataclass(frozen=True)
class Field:
    name: str
    dtype: "DType"
    metadata: Optional[MetaData] = None

    def __str__(self):
        meta = ""
        if self.metadata is not None:
            meta = (
                f"meta = {OPEN}{', '.join(f'{k}: {v}' for k,v in self.metadata)}{CLOSE}"
            )
        return f"Field('{self.name}', {str(self.dtype)}{meta})"


# -----------------------------------------------------------------------------
# Immutable Types with structural equality...


@dataclass(frozen=True)  # type: ignore
class DType(ABC):
    @property
    def size(self):
        return -1  # means unknown

    def __str__(self):
        if self.nullable:
            return f"{self.name.title()}(nullable=True)"
        else:
            return self.name

    @abstractmethod
    def constructor(self, nullable):
        pass

    def with_null(self, nullable=True):
        return self.constructor(nullable)


# for now: no float16, and all date and time stuff, categoricals, (and Null is called Void)


@dataclass(frozen=True)
class Void(DType):
    nullable: bool = True
    typecode: ClassVar[str] = "n"
    arraycode: ClassVar[str] = "b"
    name: ClassVar[str] = "void"
    default: ClassVar[Optional[bool]] = None

    @property
    def size(self):
        # currently 1 byte per bit
        return 1

    def constructor(self, nullable):
        return Void(nullable)


@dataclass(frozen=True)  # type: ignore
class Numeric(DType):
    @property
    def size(self):
        if self.name.endswith("64"):
            return 8
        if self.name.endswith("32"):
            return 4
        if self.name.endswith("16"):
            return 2
        if self.name.endswith("8"):
            return 1
        raise AssertionError("missing case")


@dataclass(frozen=True)
class Boolean(DType):
    nullable: bool = False
    typecode: ClassVar[str] = "b"
    arraycode: ClassVar[str] = "b"
    name: ClassVar[str] = "boolean"
    default: ClassVar[bool] = False

    @property
    def size(self):
        # currently 1 byte per bit
        return 1

    def constructor(self, nullable):
        return Boolean(nullable)


@dataclass(frozen=True)
class Int8(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "c"
    arraycode: ClassVar[str] = "b"
    name: ClassVar[str] = "int8"
    default: ClassVar[int] = 0

    def constructor(self, nullable):
        return Int8(nullable)


@dataclass(frozen=True)
class Uint8(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "C"
    arraycode: ClassVar[str] = "B"
    name: ClassVar[str] = "uint8"
    default: ClassVar[int] = 0

    def constructor(self, nullable):
        return Uint8(nullable)


@dataclass(frozen=True)
class Int16(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "s"
    arraycode: ClassVar[str] = "h"
    name: ClassVar[str] = "int16"
    default: ClassVar[int] = 0

    def constructor(self, nullable):
        return Int16(nullable)


@dataclass(frozen=True)
class Uint16(Numeric):
    nullable: bool = False
    typecode = "S"
    arraycode: ClassVar[str] = "h"
    name: ClassVar[str] = "uint16"
    default: ClassVar[int] = 0

    def constructor(self, nullable):
        return Uint16(nullable)


@dataclass(frozen=True)
class Int32(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "i"
    arraycode: ClassVar[str] = "i"
    name: ClassVar[str] = "int32"
    default: ClassVar[int] = 0

    def constructor(self, nullable):
        return Int32(nullable)


@dataclass(frozen=True)
class Uint32(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "I"
    arraycode: ClassVar[str] = "I"
    name: ClassVar[str] = "uint32"
    default: ClassVar[int] = 0

    def constructor(self, nullable):
        return Uint32(nullable)


@dataclass(frozen=True)
class Int64(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "l"
    arraycode: ClassVar[str] = "l"
    name: ClassVar[str] = "int64"
    default: ClassVar[int] = 0

    def constructor(self, nullable):
        return Int64(nullable)


@dataclass(frozen=True)
class Uint64(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "L"
    arraycode: ClassVar[str] = "L"
    name: ClassVar[str] = "uint64"
    default: ClassVar[int] = 0

    def constructor(self, nullable):
        return Uint64(nullable)


@dataclass(frozen=True)
class Float32(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "f"
    arraycode: ClassVar[str] = "f"
    name: ClassVar[str] = "float32"
    default: ClassVar[float] = 0.0

    def constructor(self, nullable):
        return Float32(nullable)


@dataclass(frozen=True)
class Float64(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] = "d"  # CHECK Spec ???
    arraycode: ClassVar[str] = "d"
    name: ClassVar[str] = "float64"
    default: ClassVar[float] = 0.0

    def constructor(self, nullable):
        return Float64(nullable)


@dataclass(frozen=True)
class String(DType):
    nullable: bool = False
    # no support yet for
    # fixed_size: int = -1
    typecode: ClassVar[str] = "u"  # utf8 string (n byte)
    arraycode: ClassVar[str] = "w"  # wchar_t (2 byte)
    name: ClassVar[str] = "string"
    default: ClassVar[str] = ""

    def constructor(self, nullable):
        return String(nullable)


boolean = Boolean()
int8 = Int8()
uint8 = Uint8()
int16 = Int16()
uint16 = Uint16()
int32 = Int32()
uint32 = Uint32()
int64 = Int64()
uint64 = Uint64()
float32 = Float32()
float64 = Float64()
string = String()


@dataclass(frozen=True)
class Map(DType):
    key_dtype: DType
    item_dtype: DType
    nullable: bool = False
    keys_sorted: bool = False
    name: ClassVar[str] = "Map"
    typecode: ClassVar[str] = "+m"
    arraycode: ClassVar[str] = ""

    def constructor(self, nullable):
        return Map(self.key_dtype, self.item_dtype, nullable)

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        return f"Map({self.key_dtype}, {self.item_dtype}{nullable})"


@dataclass(frozen=True)
class List_(DType):
    item_dtype: DType
    nullable: bool = False
    fixed_size: int = -1
    name: ClassVar[str] = "List_"
    typecode: ClassVar[str] = "+l"

    # ugly...
    @property
    def _dtypecode(self):
        if self.fixed_size >= 0:
            return f"+w:{self.fixed_size}"
        else:
            return "+l"

    arraycode: ClassVar[str] = ""

    def constructor(self, nullable):
        return List_(self.item_dtype, nullable)

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        fixed_size = (
            ", fixed_size=" +
            str(self.fixed_size) if self.fixed_size >= 0 else ""
        )
        return f"List_({self.item_dtype}{nullable}{fixed_size})"


@dataclass(frozen=True)
class Struct(DType):
    fields: List[Field]
    nullable: bool = False
    is_dataframe: bool = False
    metadata: Optional[MetaData] = None
    name: ClassVar[str] = "Struct"
    typecode: ClassVar[str] = "+s"
    arraycode: ClassVar[str] = ""

    def __post_init__(self):
        if self.nullable:
            for f in self.fields:
                if not f.dtype.nullable:
                    raise TypeError(
                        f"nullable structs require each field (like {f.name}) to be nullable as well."
                    )

    def constructor(self, nullable):
        return Struct(self.fields, nullable)

    def get(self, name):
        for f in self.fields:
            if f.name == name:
                return f.dtype
        raise KeyError(f"{name} not among fields")

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        flds = f"[{', '.join(str(f) for f in self.fields)}]"
        meta = ""
        if self.metadata is not None:
            meta = f", meta = {OPEN}{', '.join(f'{k}: {v}' for k,v in self.metadata)}{CLOSE}"
        if self.is_dataframe:
            return f"Schema({flds}{nullable}{meta})"
        else:
            return f"Struct({flds}{nullable}{meta})"


# only used internally for type inference


@dataclass(frozen=True)
class Tuple_(DType):
    fields: List[DType]
    nullable: bool = False
    is_dataframe: bool = False
    metadata: Optional[MetaData] = None
    name: ClassVar[str] = "Tuple"
    typecode: ClassVar[str] = "+t"
    arraycode: ClassVar[str] = ""

    def constructor(self, nullable):
        return Tuple_(self.fields, nullable)


# Schema is just a struct that is maked as standing for a dataframe


def Schema(
    fields: Optional[List[Field]] = None,
    nullable: bool = False,
    metadata: Optional[MetaData] = None,
):
    if fields is None:
        fields = []
    return Struct(fields, nullable, is_dataframe=True)


# TorchArrow does not yet support these types

# abstract
@dataclass(frozen=True)  # type: ignore
class Union_(DType):
    pass


Tag = str


@dataclass(frozen=True)  # type: ignore
class DenseUnion(DType):
    tags: List[Tag]
    name: ClassVar[str] = "DenseUnion"
    typecode: ClassVar[str] = "+ud"
    arraycode: ClassVar[str] = ""


@dataclass(frozen=True)  # type: ignore
class SparseUnion(DType):
    tags: List[Tag]
    name: ClassVar[str] = "SparseUnion"
    typecode: ClassVar[str] = "+us"
    arraycode: ClassVar[str] = ""


# Array typecodes -------------------------------------------------------------
# can be deleted once TorchArrow is implemented over velox...

# ‘b’, signed char, int, 1
# ‘B’, unsigned char, int, 1
# ‘u’, wchar_t, Unicode character, 2
# ‘h', signed short, int, 2
# ‘H’, unsigned short, int, 2
# ‘I', signed int, int, 2
# ‘I’, unsigned int, int, 2
# ‘l’, signed long, int, 4
# ‘L’, unsigned long, int, 4
# ‘q’, signed long long, int, 8,
# ‘Q’, unsigned long long, int,8
# ‘f’, float, float, 4
# ‘d’, double, float, 8


# Type test -------------------------------------------------------------------
# can be deleted once TorchArrow is implemented over velox...


def is_void(t):
    """
    Return True if value is an instance of a void type.
    """
    # print('is_boolean', t.typecode)
    return t.typecode == "n"


def is_boolean(t):
    """
    Return True if value is an instance of a boolean type.
    """
    # print('is_boolean', t.typecode)
    return t.typecode == "b"


def is_numerical(t):
    return is_integer(t) or is_floating(t)


def is_integer(t):
    """
    Return True if value is an instance of any integer type.
    """
    return t.typecode in "csilCSIL"


def is_signed_integer(t):
    """
    Return True if value is an instance of any signed integer type.
    """
    return t.typecode in "csil"


def is_unsigned_integer(t):
    """
    Return True if value is an instance of any unsigned integer type.
    """
    return t.id in "CSIL"


def is_int8(t):
    """
    Return True if value is an instance of an int8 type.
    """
    return t.typecode == "c"


def is_int16(t):
    """
    Return True if value is an instance of an int16 type.
    """
    return t.typecode == "s"


def is_int32(t):
    """
    Return True if value is an instance of an int32 type.
    """
    return t.typecode == "i"


def is_int64(t):
    """
    Return True if value is an instance of an int64 type.
    """
    return t.typecode == "l"


def is_uint8(t):
    """
    Return True if value is an instance of an uint8 type.
    """
    return t.typecode == "C"


def is_uint16(t):
    """
    Return True if value is an instance of an uint16 type.
    """
    return t.typecode == "S"


def is_uint32(t):
    """
    Return True if value is an instance of an uint32 type.
    """
    return t.typecode == "I"


def is_uint64(t):
    """
    Return True if value is an instance of an uint64 type.
    """
    return t.typecode == "L"


def is_floating(t):
    """
    Return True if value is an instance of a floating point numeric type.
    """
    return t.typecode in "fd"


def is_float32(t):
    """
    Return True if value is an instance of a float32 (single precision) type.
    """
    return t.typecode == "f"


def is_string(t):
    return t.typecode == "u"


def is_float64(t):
    """
    Return True if value is an instance of a float32 (single precision) type.
    """
    return t.typecode == "d"


def is_list(t):
    return t.typecode.startswith("+l")


def is_map(t):
    return t.typecode.startswith("+m")


def is_struct(t):
    return t.typecode.startswith("+s")


def is_primitive(t):
    return t.typecode[0] != "+"


def is_tuple(t):
    return t.typecode.startswith("+t")


# Infer types from values -----------------------------------------------------
PREFIX_LENGTH = 5


def _infer_dtype_from_value(value):
    if value is None:
        return Void()
    if isinstance(value, bool):
        return boolean
    if isinstance(value, int):
        return int64
    if isinstance(value, float):
        return float64
    if isinstance(value, str):
        return string
    if isinstance(value, list):
        dtype = infer_dtype_from_prefix(value[:PREFIX_LENGTH])
        return List_(dtype)
    if isinstance(value, dict):
        key_dtype = infer_dtype_from_prefix(list(value.keys())[:PREFIX_LENGTH])
        items_dtype = infer_dtype_from_prefix(
            list(value.values())[:PREFIX_LENGTH])
        return Map(key_dtype, items_dtype)
    if isinstance(value, tuple):
        dtypes = []
        for t in value:
            dtypes.append(_infer_dtype_from_value(t))
        return Tuple_(dtypes)


def infer_dtype_from_prefix(prefix):
    if len(prefix) == 0:
        raise ValueError(f"Cannot infer type of f{prefix}")
    dtype = _infer_dtype_from_value(prefix[0])
    for p in prefix:
        next_dtype = _infer_dtype_from_value(p)
        rtype = dtype
        dtype = _lub_dtype(dtype, next_dtype)
        # print('LUB', rtype, next_dtype, '->', dtype)
        if dtype is None:
            raise ValueError(f"Cannot infer type of f{prefix}")
    return dtype


# lub of two types for inference ----------------------------------------------


def _lub_dtype(l, r):
    if is_void(l):
        return r.with_null()
    if is_void(r):
        return l.with_null()
    if is_integer(l) and is_floating(r):
        return r.with_null(l.nullable or r.nullable)
    if is_integer(r) and is_floating(l):
        return l.with_null(l.nullable or r.nullable)
    if is_tuple(l) and is_tuple(r) and len(l.fields) == len(r.fields):
        res = []
        for i, j in zip(l.fields, r.fields):
            m = _lub_dtype(i, j)
            if m is None:
                return None
            res.append(m)
        return Tuple_(res)
    if is_map(l) and is_map(r):
        k = _lub_dtype(l.key_dtype, r.key_dtype)
        i = _lub_dtype(l.item_dtype, r.item_dtype)
        return Map(k, i) if k is not None and i is not None else None
    if is_list(l) and is_list(r):
        k = _lub_dtype(l.item_dtype, r.item_dtype)
        return List_(k) if k is not None else None
    if l.with_null() == r.with_null():
        return l if l.nullable else r
    return None


# Derive result types from 1st arg type for operators -------------------------

# DESIGN BUG: TODO Fix me later
# -- needs actually both sides, due to symetric promotion rules for //...
_arithmetic_ops = ["add", "sub", "mul", "floordiv", "truediv", "mod", "pow"]
_comparison_ops = ["eq", "ne", "lt", "gt", "le", "ge"]
_logical_ops = ["and", "or"]


def derive_dtype(left_dtype, op):
    if is_numerical(left_dtype) and op in _arithmetic_ops:
        if op == "truediv":
            return Float64(left_dtype.nullable)
        elif op == "floordiv":
            if is_integer(left_dtype):
                return Int64(left_dtype.nullable)
            else:
                return Float64(left_dtype.nullable)
        else:
            return left_dtype
    if is_boolean(left_dtype) and op in _logical_ops:
        return left_dtype
    if op in _comparison_ops:
        return Boolean(left_dtype.nullable)
    raise AssertionError(
        f"derive_dtype, unexpected type {left_dtype} for operation {op}"
    )


def derive_operator(op):
    return _operator_map[op]


def _or(a, b):
    return a or b


def _and(a, b):
    return a and b


_operator_map = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "eq": operator.eq,
    "ne": operator.ne,
    "or": _or,  # logical instead of bitwise
    "and": _and,  # logical instead of bitwise
    "floordiv": operator.floordiv,
    "truediv": operator.truediv,
    "mod": operator.mod,
    "pow": operator.pow,
    "lt": operator.lt,
    "gt": operator.gt,
    "le": operator.le,
    "ge": operator.ge,
}


def get_agg_op(op: str, dtype: DType) -> Tuple[Callable, DType]:
    if op not in _agg_ops:
        raise ValueError(f"undefined aggregation operator ({op})")
    if op in ["min", "max", "sum", "prod", "mode"]:
        return (_agg_ops[op], dtype)
    if op in ["mean", "median"]:
        return (_agg_ops[op], Float64(dtype.nullable))
    if op in ["count"]:
        return (_agg_ops[op], Int64(dtype.nullable))
    raise AssertionError('unexpected case')


_agg_ops = {
    'min': lambda c: c.min(),
    'max': lambda c: c.max(),
    'all': lambda c: c.all(),
    'any': lambda c: c.any(),
    'sum': lambda c: c.sum(),
    'prod': lambda c: c.prod(),
    'mean': lambda c: c.mean(),
    'median': lambda c: c.median(),
    'mode': lambda c: c.mode(),
    'count': lambda c: c.count(),
}

# -----------------------------------------------------------------------------
# Appendix

# Array encoding from ....
# ‘b’, signed char, int, 1
# ‘B’, unsigned char, int, 1
# ‘u’, wchar_t, Unicode character, 2
# ‘h', signed short, int, 2
# ‘H’, unsigned short, int, 2
# ‘I', signed int, int, 2
# ‘I’, unsigned int, int, 2
# ‘l’, signed long, int, 4
# ‘L’, unsigned long, int, 4
# ‘q’, signed long long, int, 8,
# ‘Q’, unsigned long long, int,8
# ‘f’, float, float, 4
# ‘d’, double, float, 8

# Arrow encoding from ....
