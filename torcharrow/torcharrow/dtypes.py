from abc import ABC, abstractmethod
import operator
from typing import ClassVar, Dict, List, Optional
from dataclasses import dataclass, field


# -----------------------------------------------------------------------------
# Aux -  needed for pretty princting 
OPEN = "{"
CLOSE = "}"


# -----------------------------------------------------------------------------
# Field, For Schema see Struct below..

MetaData = Dict[str, str]

@dataclass 
class Field:
    name: str
    dtype: "DType"
    metadata: Optional[MetaData]= None
    def __str__(self):
        meta = ""
        if self.metadata is not None:
            meta = f"meta = {OPEN}{', '.join(f'{k}: {v}' for k,v in self.metadata)}{CLOSE}" 
        return f"Field({self.name}, {str(self.dtype)}{meta})"
  


# -----------------------------------------------------------------------------
# Types -- have structural equality...

@dataclass
class DType(ABC):

    @property
    def size(self):
        return -1 # means unknown
    
    def __str__(self):
        if self.nullable:
            return f"{self.name.title()}(nullable=True)"
        else:
            return self.name

# for now: no class null, float16, and all date and time stuff 

@dataclass
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
        raise AssertionError('missing case')

@dataclass
class Boolean(DType):
    nullable: bool = False
    typecode: ClassVar[str] ='b'
    arraycode: ClassVar[str]= 'b' 
    name: ClassVar[str] = "boolean"
    default: ClassVar[bool] = False
    @property
    def size(self):
        # currently 1 byte per bit
        return 1 
@dataclass
class Int8 (Numeric):
    nullable: bool = False
    typecode: ClassVar[str] ='c'
    arraycode: ClassVar[str]= 'b'
    name: ClassVar[str] = "int8"
    default: ClassVar[int] = 0
    
@dataclass
class Uint8(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] ='C'
    arraycode: ClassVar[str]= 'B'
    name: ClassVar[str] = "uint8"
    default: ClassVar[int] = 0
@dataclass
class Int16(Numeric):
    nullable: bool = False
    typecode: ClassVar[str] ='s'
    arraycode: ClassVar[str]= 'h'
    name: ClassVar[str] = "int16"
    default: ClassVar[int] = 0
@dataclass
class Uint16 (Numeric):
    nullable: bool = False
    typecode = 'S'
    arraycode: ClassVar[str]= 'h'
    name: ClassVar[str] = "uint16"
    default: ClassVar[int] = 0
@dataclass
class Int32(Numeric):
    nullable: bool = False
    typecode : ClassVar[str]='i'
    arraycode: ClassVar[str]= 'i'
    name: ClassVar[str] = "int32"
    default: ClassVar[int] = 0
@dataclass
class Uint32 (Numeric):
    nullable: bool = False
    typecode : ClassVar[str]='I'
    arraycode: ClassVar[str]= 'I'
    name: ClassVar[str] = "uint32"
    default: ClassVar[int] = 0
@dataclass
class Int64(Numeric):
    nullable: bool = False
    typecode : ClassVar[str]= 'l'
    arraycode: ClassVar[str]= 'l'
    name: ClassVar[str] = "int64"
    default: ClassVar[int] = 0
@dataclass
class Uint64 (Numeric):
    nullable: bool = False
    typecode: ClassVar[str]= 'L'
    arraycode: ClassVar[str]= 'L'
    name: ClassVar[str] = "uint64"
    default: ClassVar[int] = 0

@dataclass
class Float32(Numeric):
    nullable: bool = False
    typecode: ClassVar[str]='f'
    arraycode: ClassVar[str] = 'f'
    name: ClassVar[str] = "float32"
    default: ClassVar[float] = 0.0
@dataclass
class Float64(Numeric):
    nullable: bool = False
    typecode: ClassVar[str]= 'd' #CHECK Spec ???
    arraycode: ClassVar[str] = 'd'
    name: ClassVar[str] = "float64"
    default: ClassVar[float] = 0.0

@dataclass
class String(DType):
    nullable: bool = False
    # no support yet for
    # fixed_size: int = -1
    typecode: ClassVar[str] = 'u'  #utf8 string (n byte)
    arraycode: ClassVar[str] = 'w' #wchar_t (2 byte)
    name: ClassVar[str] = "string"
    default: ClassVar[str] = ""
   

boolean = Boolean()
int8 =  Int8()
uint8= Uint8()
int16= Int16()
uint16= Uint16()
int32 = Int32()
uint32= Uint32 ()
int64 = Int64()
uint64 = Uint64()
float32 = Float32()
float64 = Float64()
string = String()

@dataclass
class Map(DType):
    key_dtype: DType
    item_dtype: DType
    nullable: bool = False
    keys_sorted:bool = False
    name: ClassVar[str] = "Map"
    typecode: ClassVar[str] = "+m"
    arraycode: ClassVar[str] = ""

    def __str__(self):
        nullable = ', nullable=' + str(self.nullable) if self.nullable else ""
        return f"Map({self.key_dtype}, {self.item_dtype}{nullable})"


@dataclass
class List_(DType): #
    item_dtype: DType
    nullable: bool = False
    fixed_size : int = -1
    name: ClassVar[str]= "List_"
    typecode: ClassVar[str] = "+l"
    # ugly...
    @property
    def _dtypecode(self):
        if self.fixed_size>=0:
            return f"+w:{self.fixed_size}"
        else:
           return "+l" 
    arraycode: ClassVar[str] = ""

    def __str__(self):
        nullable = ', nullable=' + str(self.nullable) if self.nullable else ""
        fixed_size = ', fixed_size=' + str(self.fixed_size) if self.fixed_size>=0 else ""
        return f"List_({self.item_dtype}{nullable}{fixed_size})"
        


@dataclass
class Struct(DType):
    fields: List[Field]
    nullable: bool = False
    is_dataframe: bool = False
    metadata: Optional[MetaData] = None
    name: ClassVar[str]= "Struct"
    typecode: ClassVar[str] = "+s"
    arraycode: ClassVar[str] = ""

    def __str__(self):
        nullable = ', nullable=' + str(self.nullable) if self.nullable else ""
        flds = f"[{', '.join(str(f) for f in self.fields)}]"
        meta = ""
        if self.metadata is not None:
            meta = f", meta = {OPEN}{', '.join(f'{k}: {v}' for k,v in self.metadata)}{CLOSE}"  
        if self.is_dataframe:
            return f"Schema({flds}{nullable}{meta})"
        else:
            return f"Struct({flds}{nullable}{meta})"

# Schema is just a struct that is maked as standing for a dataframe
def Schema( fields: Optional[List[Field]]=None, nullable: bool = False,metadata: Optional[MetaData] = None):
    if fields is None:
        fields=[]
    return Struct(fields, nullable, is_dataframe=True)


#TorchArrow does not yet support these types

#abstract
@dataclass
class Union(DType):
    pass

Tag = str
@dataclass
class DenseUnion(DType):
    tags: List[Tag]
    name: ClassVar[str] = "DenseUnion"
    typecode: ClassVar[str] = "+ud"
    arraycode: ClassVar[str] = ""

@dataclass
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


def is_boolean(t):
    """
    Return True if value is an instance of a boolean type.
    """
    return t.typecode =='b'

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
    return t.typecode =='c'


def is_int16(t):
    """
    Return True if value is an instance of an int16 type.
    """
    return t.typecode =='s'


def is_int32(t):
    """
    Return True if value is an instance of an int32 type.
    """
    return t.typecode =='i'


def is_int64(t):
    """
    Return True if value is an instance of an int64 type.
    """
    return t.typecode =='l'


def is_uint8(t):
    """
    Return True if value is an instance of an uint8 type.
    """
    return t.typecode =='C'


def is_uint16(t):
    """
    Return True if value is an instance of an uint16 type.
    """
    return t.typecode =='S'


def is_uint32(t):
    """
    Return True if value is an instance of an uint32 type.
    """
    return t.typecode =='I'


def is_uint64(t):
    """
    Return True if value is an instance of an uint64 type.
    """
    return t.typecode =='L'


def is_floating(t):
    """
    Return True if value is an instance of a floating point numeric type.
    """
    return t.typecode in "fd"


def is_float32(t):
    """
    Return True if value is an instance of a float32 (single precision) type.
    """
    return t.typecode =='f'


def is_string(t):
    return t.typecode=="u"

def is_float64(t):
    """
    Return True if value is an instance of a float32 (single precision) type.
    """
    return t.typecode =='d'

def is_list(t):
    return t.typecode.startswith("+l")

def is_map(t):
    return t.typecode.startswith("+m")

def is_struct(t):
    return t.typecode.startswith("+s")

def is_primitive(t):
    return t.typecode[0]!="+"



# Infer types from values -----------------------------------------------------

def infer_dtypecode(value):
    if value is None:
        return 'n'
    if isinstance(value, bool):
        return 'b'
    if isinstance(value, int):
        return 'L'
    if isinstance(value, float):
        return 'd'
    if isinstance(value, str):
        return 'u'

    return "" # could not infer typecode


def infer_dtype(prefix):
    if len(prefix)==0:
         raise ValueError(f'Cannot infer type of f{prefix}')
    # only flat lists, no recursion
    tc = None
    nullable = False
    for p in prefix:
        tc_p = infer_dtypecode(p)
        if "n" in tc_p:
            nullable=True
            continue
        if tc is None:
            tc = tc_p
        elif tc == tc_p:
            continue
        elif tc == 'd' and tc_p == 'L': 
            continue
        elif tc == 'L' and tc_p == 'd':
            # promotion of int to float...
            tc= 'd'
            continue
        else:
            raise ValueError(f"Different types can't be used within one column: {prefix}")

    if tc=="d": return Float64(nullable)
    if tc=="b": return Boolean(nullable)
    if tc=="L": return Int64(nullable)
    if tc=="u": return String(nullable)
    raise NotImplementedError(f"Inference for {prefix} is still missing")



# Derive result types from 1st arg type for operators -------------------------

def derive_dtype(left_dtype, op):
    if is_numerical(left_dtype) and op in arithmetic_ops:
        return left_dtype
    if is_string(left_dtype) and op in ["add"]:
        return left_dtype
    if is_boolean(left_dtype) and op in ["and,or"]:
        return left_dtype
    if op in comparison_ops:
        return boolean
    if op == 'in':
        return boolean

def derive_operator(op):
    return operator_map[op]

arithmetic_ops = [
    "add",
    "sub",
    "mul",
    "floordiv",
    "truediv",
    "mod" , 
    "pow"]
comparison_ops = [
    "eq",   
    "ne", 
    "lt", 
    "gt", 
    "le", 
    "ge"]

def or_(a,b): return a or b
def and_(a,b): return a and b
def contains_(obj, seq): return obj in seq


operator_map = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,    
    "eq": operator.eq,   
    "ne": operator.ne , 
    "or": or_ , # logical instead of bitwise
    "and": and_, # logical instead of bitwise
    "floordiv": operator.floordiv,
    "truediv": operator.truediv,
    "mod": operator.mod , 
    "pow": operator.pow , 
    "lt": operator.lt , 
    "gt": operator.gt , 
    "le": operator.le, 
    "ge": operator.ge,
    "in": contains_}



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
