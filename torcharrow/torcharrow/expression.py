import operator
from abc import ABC
from keyword import iskeyword
from os import stat
from typing import Any, Callable, Mapping, Optional, Sequence

# -----------------------------------------------------------------------------
# Expressions


class Expression(ABC):
    """ 
    Abstract class representing Python's parsed expressions trees but without if expressions
    """

    def __getattr__(self, name):
        """Construct a symbolic representation of `getattr(self, name)`."""
        return GetAttr(self, name)

    def __call__(self, *args, **kwargs):
        """Construct a symbolic representation of `self(*args, **kwargs)`."""
        return Call(self, args, kwargs)

# function decorator ----------------------------------------------------------


def expression(fn):
    """Register function as available as part of expression."""
    # TODO should we hijack this (or have yet another doc decorator)
    # to also print a summary of all functions? Yes:-)
    # print('function', fn.__name__)
    setattr(Expression, fn.__name__, lambda self, *args, **
            kwargs: Call(GetAttr(self, fn.__name__), args, kwargs))

    return fn

# subtypes --------------------------------------------------------------------


class Var(Expression):
    def __init__(self, id: str):
        self._id = id
        # print("VAR", self)

    def eval_expression(self, env):
        return env[self._id]

    def __str__(self):
        return f"{self._id}"


class GetAttr(Expression):

    def __init__(self, obj: Any, name: str):
        self._obj = obj
        self._name = name
        # print(f'GEATTR: {self}')

    def eval_expression(self, env):
        evaled_obj = eval_expression(self._obj, env)
        return getattr(evaled_obj, self._name)

    def __str__(self):
        return f"{self._obj}.{self._name}"


def _dummy_function():
    pass


class Call(Expression):

    def __init__(self,  _func: Callable, _args: Optional[Sequence] = None, _kwargs: Optional[Mapping] = None):
        self._func = _func
        self._args = _args
        self._kwargs = _kwargs
        # print(
        #     f'CALL: fun= {self._func}, args= {self._args}, kwargs= {self._kwargs}')

    def eval_expression(self, env):
        evaled_func = eval_expression(self._func, env)
        evaled_args = []
        if self._args is not None:
            evaled_args = [eval_expression(v, env) for v in self._args]
        evaled_kwargs = {}
        if self._kwargs is not None:
            evaled_kwargs = {k: eval_expression(v, env)
                             for k, v in self._kwargs.items()}
        return evaled_func(*evaled_args, **evaled_kwargs)

    def _str(v):
        if isinstance(v, (int, float, bool)):
            return str(v)
        if isinstance(v, str):
            return "'" + v + "'"
        if isinstance(v, tuple):
            return "(" + ', '.join(Call._str(w) for w in v) + ',)'
        if isinstance(v, Sequence):
            return "[" + ', '.join(Call._str(w) for w in v) + ']'
        if isinstance(v, Mapping):
            return "{" + ', '.join(Call._str(k) + ': ' + Call._str(w) for k, w in v.items()) + '}'
        if isinstance(v, type(_dummy_function)):
            return v.__qualname__
        if isinstance(v, type(operator.add)):
            return "operator." + v.__name__
        return str(v)

    def __str__(self):
        args = []
        # print("STR", self._func, 'args=', self._args, 'kwargs=', self._kwargs)
        if self._args is not None:
            args = [Call._str(v) for v in self._args]
        if self._kwargs is not None:
            if all(k.isidentifier() and not iskeyword(k) for k in self._kwargs.keys()):
                args += [f'{str(k)}={Call._str(v)}' for k,
                         v in self._kwargs.items()]
            else:
                args += "**{" + ', '.join(f'{k} : {Call._str(v)}' for k,
                                          v in self._kwargs.items()) + "}"
        if isinstance(self._func, type(_dummy_function)) and self._func.__name__ == "__init__":
            # from T.__init(self,....) to T(...)
            qname = self._func.__qualname__
            return f"{qname[:qname.rindex('.')]}({', '.join(args[1:])})"
        if isinstance(self._func, type(_dummy_function)) and hasattr(self._func, "_is_property"):
            return f'{args[0]}.{self._func.__name__}'
        if isinstance(self._func, type(_dummy_function)):
            return f"{self._func.__qualname__}({', '.join(args)})"
        if isinstance(self._func, GetAttr):
            return f"{self._func}({', '.join(args)})"
        raise AssertionError(
            f'unexpected case {type(self._func)} {self._func}')


# subtypes --------------------------------------------------------------------

def eval_expression(obj, env):
    return obj.eval_expression(env) if hasattr(obj, 'eval_expression') else obj
