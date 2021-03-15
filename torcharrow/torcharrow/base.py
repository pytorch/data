# import array as ar
# from abc import ABC

# from typing import Union

# # -----------------------------------------------------------------------------
# # Buffer types...
# NullBuffer = ar.array
# DataBuffer = ar.array
# OffsetBuffer = ar.array

# # -----------------------------------------------------------------------------
# # Aux -  needed for pretty princting 
# OPEN = "{"
# CLOSE = "}"
# NL="\n"

# ScalarTypeValues = (int, float, bool, str)

# ScalarTypes= Union[int,float,bool,str]

# def is_scalar(x):
#     return isinstance(x, ScalarTypeValues)


# ------------------------------------------------------------------------------
# AbstractColumn - the abstract baseclass

# factory = {}

# def set_constructor(dtype, constructor):
#     factory[normalize(dtype)] = constructor

# def get_constructor(dtype):

#     return factory[dtype]

# class AbstractColumn(ABC):
    
#     def __init__(self):
#         # initialized by each concrete Colunn class 
#         # -- needed to break cyclic dependency enbtween base and sub classes 
#         self._constructor = None

#     def _set_constructor(constructor):
#         self._constructor = constructor


    



# set_constructor(int8, NumericalColumn(int8))
        