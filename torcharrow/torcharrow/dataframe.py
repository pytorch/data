import operator
from os import stat
import statistics
import math

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence, Type, Union, Optional, List, Dict, ClassVar, Callable, Any
from tabulate import tabulate

# low level stuff -- all vectors are in arrays of primitive type
import array as ar

# high level stuff -- all types are described here
from .dtypes import (DType, MetaData, derive_operator, String, boolean, Int64, Boolean, List_, Struct,
is_numerical, is_string,  is_boolean, is_primitive, is_map, is_struct, is_list, infer_dtype, 
derive_dtype, derive_operator, Field, Schema)

# -----------------------------------------------------------------------------
# Buffer types...
NullBuffer = ar.array
DataBuffer = ar.array
OffsetBuffer = ar.array

# -----------------------------------------------------------------------------
# Aux -  needed for pretty princting 
OPEN = "{"
CLOSE = "}"
NL="\n"


# ------------------------------------------------------------------------------
# _Column - the astract baseclass; even dataframes are just struct columns

@dataclass
class _Column(ABC):
    _dtype: DType
    _count: int = 0
    _null_count:int = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        # count is used on strings and lists already
        return self._count
  
    def __len__(self):
        return self._count

    def __repr__(self):
        return self.__str__()

    # constructor factory -----------------------------------------------------



    # selectors ------------------------------------------------------------

    @abstractmethod
    def get(self, i, fill_value):
        pass
    
    def __getitem__(self, i):
        # _lookup by string
        # TODO maybe check that the subclass has actually a _lookup.
        if isinstance(i, str):
            return self._lookup(i)
        if isinstance(i, (list,tuple)) and all(isinstance(k, str) for k in i):
            return tuple( self._lookup(k) for  k in i)
        # _lookup by int and slice
        if isinstance(i, int):
            return self.get(i, None)
        if isinstance(i, slice):
            # TODO generalize this...
            assert i.start is not None and i.stop is not None
            # TODO fix this: e.g. by creating a new columns instead
            # res = _create_column(self._dtype)
            # for j in range(i.start, i.stop):
            #     res.append( self.get(j, None))
            # return res
            # # but this requires  changes in tests, so skip for now!
            return [ self.get(j, None) for j in range(i.start, i.stop)] 

        # filter/where
        if isinstance(i, _BooleanColumn):
            res = _create_column(self._dtype)
            for x,m in zip(list(self), list(i)):
                if m:
                    res.append(x)
            return res
        # raise AssertionError('unexpected case')

    def slice(self, start=None, stop=None, step=1):
        if (step < 1):
            raise ValueError(f"step must be > 0. given: {step}")
        res = _create_column(self._dtype)
        for j in range(start, stop, step):
            res.append( self.get(j, None))
        return res



    # builders ------------------------------------------------------------

    @abstractmethod
    def append(self, value):
        pass

    def extend(self, iterable):
        for i in iterable:
            self.append(i)

    # list ----------------------------------------------------------------
    @abstractmethod 
    def __iter__(self):
        while False:
            yield None

    def to_list(self):
        return list(self.__iter__())

    # binary operators --------------------------------------------------------

    def _broadcast(self, operator, const, dtype):
        assert is_primitive(self._dtype) and is_primitive(dtype) 
        res = _create_column(dtype)
        for i in range(self._count):
            if self._validity[i]:
                res.append(operator( self._data[i], const))
            else:
                res.append(None)   
        return res

    def _pointwise(self, operator, other, dtype):
        assert is_primitive(self._dtype) and is_primitive(dtype) 
        res = _create_column(dtype)
        for i in range(self._count):
            if self._validity[i] and other._validity[i]:
                res.append(operator( self._data[i], other._data[i]))
            else:
                res.append(None)   
        return res
           

    def _binary_operator(self, operator, other):
        if isinstance(other, (int, float, list, set, type(None))):  
            return self._broadcast(derive_operator(operator), other,  derive_dtype(self._dtype, operator))
        else:
            return self._pointwise( derive_operator(operator), other,  derive_dtype(self._dtype, operator))

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

    # other ops ------------------------------------------------------------

    def fillna(self,fill_value):
        assert fill_value is not None
        res = _create_column(self._dtype)
        for i in range(self._count):
            if self._validity[i]:
                res.append(self[i])
            else:
                res.append(fill_value)   
        return res
        
    def dropna(self):
        res = _create_column(self._dtype)
        for i in range(self._count):
            if self._validity[i]:
                res.append(self[i])
        return res

    def head(self, n):
        res = _create_column(self._dtype)
        for i in range(0,n):
            if self._validity[i]:
                res.append(self[i])
            else:
                res.append(None)
        return res

    def min(self):
        return min(list(self))
    
    def max(self):
        return max(list(self))
    
    # fnuctools ---------------------------------------------------------------

    def map(self, fun, dtype:Optional[DType]=None):
        #dtype must be given, if result is different from argument column
        if dtype is None:
            dtype = self._dtype
        res = _create_column(dtype)
        for i in range(self._count):
            if self._validity[i]:
                res.append(fun(self[i]))
            else:
                res.append(None)
        return res
        
    def filter(self, pred):
        res = _create_column(self._dtype)
        for i in range(self._count):
            if self._validity[i]:
                if pred(self[i]):
                    res.append(self[i])
                    continue          
            else:
                res.append(None)
        return res

    def reduce(self, fun, initializer=None):
        if self._count==0:
            if initializer is not None:
                return initializer
            else:
                raise TypeError("reduce of empty sequence with no initial value")
        start = 0
        if initializer is None:
            value = self[0]
            start = 1
        else: 
            value = initializer
        for i in range(start,self._count):
            value = fun(value, self[i])
        return value

    def flatmap(self, fun, dtype:Optional[DType]=None):
        #dtype must be given, if result is different from argument column
        if dtype is None:
            dtype = self._dtype
        res = _create_column(dtype)
        for i in range(self._count):
            if self._validity[i]:
                res.extend(fun(self[i]))
            else:
                res.append(None)
        return res
        

# ------------------------------------------------------------------------------
# _Column

class _NumericalColumn(_Column):

    def __init__(self, dtype):
        assert is_numerical(dtype)

        super().__init__(dtype)
        self._data = ar.array(dtype.arraycode)
        self._validity = ar.array('b')
        self._null_count = 0
        self._offset = 0
        self._count = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self._null_count> 0:
            tab = tabulate([[l,v] for (l,v) in zip(self._data, self._validity)],['data', 'validity'])
        else:
            tab = tabulate([[l] for l in self._data],['data'])
        typ = f"dtype: {self._dtype}, count: {self._count}, null_count: {self._null_count}" 
        return tab+NL+typ
          
    def append(self, x):
        if x is None and self._dtype.nullable:
            self._data.append(self._dtype.default)
            self._validity.append(False)
            self._null_count += 1
        else:
            self._data.append(x)
            self._validity.append(True)
        self._count += 1


    def get(self, i, fill_value):
        if self._null_count==0:
            return self._data[i]
        elif not self._validity[i]:
            return fill_value
        else:
            return self._data[i]

    def __iter__(self):
        if self._null_count==0:
            for x in self._data:
                yield x
        else:
            for x,n in zip(self._data, self._validity):
                if not n:
                    yield None
                else:
                    yield x

    # descriptive statistics --------------------------------------------------
    # special operations on Numerical Columns

    def sum(self):
        if self._null_count==0:
            return sum(self._data)
        else:
            return sum(self._data[i] 
                        for i in range(self._count) 
                        if self._validity[i])

    def mean(self):
        if self._null_count==0:
            return statistics.mean(self._data)
        else:
            return statistics.mean(self._data[i] 
                        for i in range(self._count) 
                        if self._validity[i])

# ------------------------------------------------------------------------------
# _Column

class _BooleanColumn(_Column):

    def __init__(self, dtype):
        super().__init__(dtype)
        self._data = ar.array('b') 
        self._validity = ar.array('b')
        self._null_count = 0
        self._offset = 0
        self._count = 0

    def __str__(self):
        if self._null_count> 0:
            tab = tabulate([[bool(l),v] for (l,v) in zip(self._data, self._validity)],['data', 'validity'])
        else:
            tab = tabulate([[bool(l)] for l in self._data],['data'])
        typ = f"dtype: {self._dtype}, count: {self._count}, null_count: {self._null_count}" 
        return tab+NL+typ
            
    def append(self, x):
        if x is None and self._dtype.nullable:
            self._data.append(self._dtype.default)
            self._validity.append(False)
            self._null_count += 1
        else:
            self._data.append(x)
            self._validity.append(True)
        self._count += 1



    def get(self, i, fill_value):
        if self._null_count==0:
            return bool(self._data[i])
        elif not self._validity[i]:
            return fill_value
        else:
            return bool(self._data[i])

    def __iter__(self):
        if self._null_count==0:
            for x in self._data:
                yield bool(x)
        else:
            for x,n in zip(self._data, self._validity):
                if not n:
                    yield None
                else:
                    yield bool(x)


# ------------------------------------------------------------------------------
# _StringColumn

class _StringColumn(_Column):
    # private

    def __init__(self,  dtype):
        super().__init__(dtype)
        # for simplicty always add a null validity
        self._validity = ar.array('b')
        self._offsets = ar.array('I', [0])
        self._data = ar.array('u') 
        self._offset = 0
        self._null_count = 0
        self._count =0

    def __str__(self):
            if self._null_count> 0:
                tab = tabulate([ [self._data[self._offsets[i]:
                                  self._offsets[i+1]].tounicode(),o,v] 
                                 for i,(o,v) in enumerate(zip(self._offsets, self._validity))],
                                 ['data', 'offsets', 'validity'])
            else:
                tab = tabulate([ [self._data[self._offsets[i]:
                                  self._offsets[i+1]].tounicode(),o] 
                                 for i,(o,v) in enumerate(zip(self._offsets, self._validity))],
                                 ['data', 'offsets'])
            typ = f"dtype: {self._dtype}, count: {self._count}, null_count: {self._null_count}" 
            return tab+NL+typ

    def append(self, cs):
        if cs is None:
            self._null_count += 1
            self._validity.append(False)
            self._offsets.append(self._offsets[-1])
        else:
            self._validity.append(True)
            self._data.extend(cs)
            self._offsets.append(self._offsets[-1]+len(cs))
        self._count += 1
            

    def get(self, i, fill_value):
        if self._null_count==0:
            return self._data[self._offsets[i]:self._offsets[i+1]].tounicode()
        elif not self._validity[i]:
            return fill_value
        else:
            return self._data[self._offsets[i]:self._offsets[i+1]].tounicode()


    def __iter__(self):
        for i in range(self._count):
            if self._validity[i]:
                yield self._data[self._offsets[i]:self._offsets[i+1]].tounicode()
            else:
                yield None


    # string ops --------------------------------------------------------------
    # special operations on String Columns

    def count(self, x):     
        "total number of occurrences of x in s"
        res = _NumericalColumn(Int64(self._dtype.nullable))
        for i in self.iter():
            res.append(i._count(x))
        return res

    def capitalize(self):
        res = _StringColumn(String(self._dtype.nullable))
        for i in range(self._count):
            if self._validity[i]:
                res.append(self[i].capitalize())
            else:
                 res.append(None)
        return res

    def startswith(self, x):
        res = _BooleanColumn(String(self._dtype.nullable))
        for i in self.iter():
            res.append(i.startswith(x))
        return res
    
# ops on strings --------------------------------------------------------------
#  'capitalize',
#  'casefold',
#  'center',
#  'count',
#  'encode',
#  'endswith',
#  'expandtabs',
#  'find',
#  'format',
#  'format_map',
#  'offset',
#  'isalnum',
#  'isalpha',
#  'isascii',
#  'isdecimal',
#  'isdigit',
#  'isidentifier',
#  'islower',
#  'isnumeric',
#  'isprintable',
#  'isspace',
#  'istitle',
#  'isupper',
#  'join',
#  'ljust',
#  'lower',
#  'lstrip',
#  'maketrans',
#  'partition',
#  'removeprefix',
#  'removesuffix',
#  'replace',
#  'rfind',
#  'rindex',
#  'rjust',
#  'rpartition',
#  'rsplit',
#  'rstrip',
#  'split',
#  'splitlines',
#  'startswith',
#  'strip',
#  'swapcase',
#  'title',
#  'translate',
#  'upper',
#  'zfill']

# -----------------------------------------------------------------------------
# List

class _ListColumn(_Column):

    def __init__(self,  dtype, item_col: _Column):
        super().__init__(dtype)
        # TODO: Check offset logic ... for now assume
        assert(len(item_col)==0)
        self._offsets = ar.array('I', [0])  
        self._item_data = item_col
        self._offset = 0
        self._null_count = 0
        self._count=len(item_col)
        self._validity = ar.array('b',[True]*self._count)
    
    def __str__(self):
            if self._null_count> 0:
                tab = tabulate([ [self._item_data[self._offsets[i]:
                                  self._offsets[i+1]],o,v] 
                                 for i,(o,v) in enumerate(zip(self._offsets, self._validity))],
                                 ['data', 'offsets', 'validity'])
            else:
                tab = tabulate([ [self._item_data[self._offsets[i]:
                                  self._offsets[i+1]],o] 
                                 for i,(o,v) in enumerate(zip(self._offsets, self._validity))],
                                 ['data', 'offsets'])
            typ = f"dtype: {self._dtype}, count: {self._count}, null_count: {self._null_count}" 
            return tab+NL+typ

    def append(self, values):        
        if values is None:
            self._null_count += 1
            self._validity.append(False)
            self._offsets.append(self._offsets[-1])
        else:
            self._validity.append(True)
            self._item_data.extend(values)
            self._offsets.append(self._item_data._count)
        self._count += 1


    def get(self, i, fill_value):
        if self._null_count==0:
            return self._item_data[self._offsets[i]:self._offsets[i+1]]
        elif not self._validity[i]:
            return fill_value
        else:
            return self._item_data[self._offsets[i]:self._offsets[i+1]]


    def __iter__(self):
        for i in range(self._count):
            if self._validity[i]:
                yield self._item_data[self._offsets[i]:self._offsets[i+1]]
            else:
                yield None


    # list_ops-----------------------------------------------------------------

    def count(self, x):     
        "total number of occurrences of x in s"
        res = _NumericalColumn(Int64(self._dtype.nullable))
        for i in self.iter():
            res.append(i._count(x))
        return res

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
    def __init__(self,  dtype, key_col: _Column, item_col: _Column):
        super().__init__(dtype)
        # TODO: Check offset logic, in caxe you pass an existsing column... for now assume
        assert(len(item_col)==0)
        self._offsets = ar.array('I', [0])
        self._key_data = key_col
        self._item_data = item_col
        self._offset = 0
        self._null_count = 0
        if len(key_col) != len(item_col):
                raise TypeError("key and item columns must have eual length")
        self ._count = len(key_col)
        self._validity = ar.array('b',[True]*self._count)

    
    def __str__(self):
        return f"_MapColumn[{id(self)}]({self._dtype}, {self._count}, {self._null_count}, {self._offset}, {self._offsets}, {self._validity}, ID{id(self._key_data)}, ID{id(self._key_data)})"


    def append(self, map):        
        if map is None:
            self._null_count += 1
            self._validity.append(False)
            self._offsets.append(self._offsets[-1])
        else:
            self._validity.append(True)
            self._key_data.extend([k for k in map.keys()])
            self._item_data.extend([v for v in map.values()]) 
            self._offsets.append(self._key_data._count)
        self._count += 1

    def get(self, i, fill_value):
        if self._null_count==0:
            return {self._key_data[i] : self._item_data[i] 
                    for i in range(self._offsets[i], self._offsets[i+1])}
        elif not self._validity[i]:
            return fill_value
        else:
            return {self._key_data[i] : self._item_data[i] 
                    for i in range(self._offsets[i], self._offsets[i+1])}


    def __iter__(self):
        for i in range(self._count):
            if self._validity[i]:
                yield {self._key_data[i] : self._item_data[i] 
                    for i in range(self._offsets[i], self._offsets[i+1])}
            else:
                yield None

    
        
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
# Struct


class _StructColumn(_Column):

    internal_fields = ["_dtype", "_null_count", "_offset", "_count", "_validity","_field_data"]

    def __init__(self,  dtype, field_col: Dict[str,_Column]):
        super().__init__(dtype)
        if  set(f.name for f in dtype.fields) != set(field_col.keys()):
            raise TypeError("type and columns ust match")
        #invariant, field_data order is dtype_order
        self._field_data = {}
        for f in dtype.fields:
            self._field_data[f.name]= field_col[f.name]
            if f.name in _StructColumn.internal_fields:
                 raise TypeError("column name cannot be one of '{_StructColumn.internal_fields}'")

        self._offset = 0
        self._null_count = 0
        self._count = -1
        for k in field_col.keys():
            if  self._count == -1:
                self._count = len(field_col[k])
                continue
            if self._count != len(field_col[k]):
                raise TypeError("all columns must have eual length")
        self._count = max(0,self._count )
        self._validity = ar.array('b',[True]*self._count)
        
    def __str__(self):
            cols = {f:list(c) for f,c in self._field_data.items()}
            if self._null_count> 0:
                cols["validity"]= list(self._validity)
                tab = tabulate(cols,self._field_data.keys()+["validity"])
            else:
                tab = tabulate(cols,self._field_data.keys())
            typ = f"dtype: {self._dtype}, count: {self._count}, null_count: {self._null_count}" 
            return tab+NL+typ

    def _lookup(self,name):
        return self._field_data[name]
    
    def columns(self):
        return self._field_data

    # for now struct values are just tuples
    # TODO add named tuples, dataclasses
    def append(self, tup):        
        if tup is None:
            self._null_count += 1
            self._validity.append(False)
            self._offsets.append(self._offsets[-1])
        else:
            assert isinstance(tup, tuple)
            self._validity.append(True)
            for i,v in enumerate(tup):
                self._field_data[self._dtype.fields[i].name].append(v)
        self._count += 1

    def get(self, i, fill_value):
        if self._null_count==0:
            return tuple(self._field_data[f.name][i] for f in self._dtype.fields)
        elif not self._validity[i]:
            return fill_value
        else:
            return tuple(self._field_data[f.name][i] for f in self._dtype.fields)


    def __iter__(self):
        for i in range(self._count):
            if self._validity[i]:
                yield tuple(self._field_data[f.name][i] for f in self._dtype.fields)
            else:
                yield None

    # project/rename --------------------------------------------------------------
    def drop(self,col_names):
        flds = []
        cols = {}
        for f in self._dtype.fields:
            if f.name not in col_names:
                flds.append(f)
                cols[f.name] = self._field_data[f.name]
        return _StructColumn(Struct(flds, self._dtype.nullable, self._dtype.is_dataframe), cols)

    def keep(self,col_names):
        flds = []
        cols = {}
        for f in self._dtype.fields:
            if f.name in col_names:
                flds.append(f)
                cols[f.name] = self._field_data[f.name]
        return _StructColumn(Struct(flds, self._dtype.nullable, self._dtype.is_dataframe), cols)

    def rename(self,map):
        assert isinstance(map, dict) and all(isinstance(k, str) and isinstance(v, str) for k,v in map.items())
        flds = []
        cols = {}
        for f in self._dtype.fields:
            if f.name in map:
                flds.append(Field(map[f.name], f._dtype))
                cols[f.name] = self._field_data[f.name]
        return _StructColumn(Struct(flds, self._dtype.nullable, self._dtype.is_dataframe), cols)


    # getter/setter--------------------------------------------------------------
    # TODO CLEAN Up setter.getter 

    def __setitem__(self, name: str, value: Any) -> None:
        if name in self.__dict__ or name in _StructColumn.internal_fields :
            self.__dict__[name] = value
            return 
       
        if not isinstance(value, _Column):
            raise TypeError(f"only columns can be updated {name} {value}")
       
        if len(self._dtype.fields)==0:
            # first column added
            self._dtype.fields.append(Field(name, value._dtype))
            self._field_data[name]=value
            self._count=len(value)
            self._null_count=0
            self._validity = ar.array('b',[True]*self._count)
            return

        if len(value) != self._count:
            raise TypeError("all columns must have same length")
       
        if name in self._field_data:
            # update the fields type and column.
            self._field_data[name]=value
            for f in self._dtype.fields:
                if f.name == name:
                    f.type = value._dtype
                    return # early
        else:
            # add the column (add the end)
            self._dtype.fields.append(Field(name, value._dtype))
            self._field_data[name]=value

    

    def __getitem__(self, name) -> None:
        if isinstance(name,str) and (name in self.__dict__ or name in _StructColumn.internal_fields) :
             return self.__dict__[name]
        if isinstance(name, (str, tuple, slice, int, _BooleanColumn)):
            return super().__getitem__(name)
        else:
            try:
                return self._field_data[name]
            except KeyError:
                raise AttributeError(f"{type(self)} object has no attribute '{name}'") from None

    # def __setitem__(self, name: str, value: Any) -> None:
    #     self.__setattr__(name,value)



# -----------------------------------------------------------------------------
# Column and Dataframe factories.
# -- note that Dataframe is in qutessence an alias for a _StructColumn 

# public factory API

def DataFrame(initializer: Union[Dict, DType, None]= None, dtype: Optional[DType]=None) -> _Column:
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
            cols= {}
            fields=[]
            for k,vs in initializer.items():
                cols[k] = Column(vs)
                fields.append(Field(k,cols[k]._dtype))
            return _StructColumn(Schema(fields), cols)
        else:
            raise ValueError('cannot infer type of initializer')
    


def Column(initializer: Union[Dict, List, DType, None]= None, dtype: Optional[DType]=None) -> _Column:
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
            cols= {}
            fields=[]
            for k,vs in initializer.items():
                cols[k] = Column(vs)
                fields.append(Field(k,cols[k]._dtype))
            return _StructColumn(Struct(fields), cols)
    elif isinstance(initializer, list):
        dtype = infer_dtype(initializer[0:5])
        if dtype is None:
            raise ValueError('cannot infer type of initializer')
        col = _create_column(dtype)
        for i in initializer:
            col.append(i)
        return col
    else:
        raise ValueError('cannot infer type of initializer')

    
@staticmethod
def arange(
        start: Union[int, float],
        stop: Union[int, float,None] = None,
        step: Union[int, float] = 1,
        dtype:Optional[DType]=None) -> "_Column":
        return Column(list(range(start,stop,step)),dtype)

# private factory method
def _create_column(dtype: Optional[DType] = None):
        if dtype is None:
            return _StructColumn(Struct([]))
        if is_numerical(dtype):
            return _NumericalColumn(dtype)
        if is_string(dtype):
            return _StringColumn(dtype) 
        if is_boolean(dtype):
            return _BooleanColumn(dtype)
        if is_list(dtype):
            return _ListColumn(dtype, _create_column(dtype.item_dtype))
        if is_map(dtype):
            return _MapColumn(dtype, 
                _create_column(dtype.key_dtype), 
                _create_column(dtype.item_dtype))
        if is_struct(dtype):
            return _StructColumn(dtype, {f.name: _create_column(f.dtype) for f in dtype.fields})
        raise AssertionError(f'unexpected case: {dtype}')


