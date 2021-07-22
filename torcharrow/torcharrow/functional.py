from types import ModuleType
from typing import Callable, Dict, List, Optional
import torcharrow.dtypes as dt
from dataclasses import dataclass


# type of functions in functional. Having this class makes it easy to
# to differentiate functions in functional from regular Callable objects (python functions for example)
# and also make it easy to access udf_name and alias from function object
@dataclass(frozen=True)
class FunctionHandle:
    udf_name: str
    alias: str
    fn: Callable

    def __call__(self, *args):
        return self.fn(*args)


@dataclass(frozen=True)
class FunctionSignature:
    arg_types: List[dt.DType]
    return_type: Optional[dt.DType]
    help_msg: str


@dataclass(frozen=True)
class GenericUDF:
    fn: FunctionHandle
    signatures: List[FunctionSignature]


class _Namespace(ModuleType):
    """Exposes C++ UDFs from registry as a python module members."""

    def __init__(self, name: str):
        super().__init__("torcharrow.functional." + name)
        self._registered_functions: Dict[str, GenericUDF] = {}

    def register_function(self, udf_name: str, fn: Callable, signatures: List[FunctionSignature], alias: Optional[str] = None):
        alias = alias or udf_name
        assert alias not in dir(self)
        #TODO: fn.__doc__ = ';'.join(signature.help_msg for signature in signatures)
        function = FunctionHandle(udf_name, alias, fn)
        self._registered_functions[alias] = GenericUDF(function, signatures)
        setattr(self, alias, function)

    def __getattr__(self, key: str) -> FunctionHandle:
        if (fn := self.__dict__.get(key)) is None:
            raise TypeError(f"'{key}' is not a registered UDF in '{self.__name__}'.")
        return fn

    @property
    def registered_functions(self) -> Dict[str, GenericUDF]:
        return self._registered_functions


class _Functional(ModuleType):
    def __init__(self):
        super().__init__('torcharrow.functional')
        self._namespaces: Dict[str, _Namespace] = {}

    def __getattr__(self, name: str) -> _Namespace:
        if (namespace := self._namespaces.get(name)) is None:
            namespace = _Namespace(name)
            self._namespaces[name] = namespace
        return namespace


functional = _Functional()
