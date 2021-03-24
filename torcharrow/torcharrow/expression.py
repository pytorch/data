import operator
import unittest
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any, List, Optional, Callable, Sequence, Mapping

# import numpy as np
# import pandas as pd
# import pyarrow as pa

from torcharrow import DataFrame

# -----------------------------------------------------------------------------
# Adopted from: https://github.com/coursera/pandas-ply


# -----------------------------------------------------------------------------
# Expressions


class Expression(ABC):
    def __getattr__(self, name):
        """Construct a symbolic representation of `getattr(self, name)`."""
        return GetAttr(self, name)

    def __call__(self, *args, **kwargs):
        """Construct a symbolic representation of `self(*args, **kwargs)`."""
        return Call(self, args=args, kwargs=kwargs)

    # # needed for testing min
    # def __iter__(self):
    #     yield 1

    # # needed for testing min
    # def __len__(self):
    #     return Call(self, args=[], kwargs={})

    # # needed for testing min
    # def __index__(self):
    #     return 1


# adding all magic names to EXpression-----------------------------------------


_magic_method_names = [
    '__abs__', '__add__', '__and__', '__cmp__', '__complex__', '__contains__',
    '__delattr__', '__delete__', '__delitem__', '__delslice__', '__div__',
    '__divmod__', '__enter__', '__eq__', '__exit__', '__float__',
    '__floordiv__', '__ge__', '__get__', '__getitem__', '__getslice__',
    '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__', '__idiv__',
    '__ifloordiv__', '__ilshift__', '__imod__', '__imul__', '__index__',
    '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__',
    '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__long__',
    '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__',
    '__nonzero__', '__oct__', '__or__', '__pos__', '__pow__', '__radd__',
    '__rand__', '__rcmp__', '__rdiv__', '__rdivmod__', '__repr__',
    '__reversed__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__',
    '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
    '__rtruediv__', '__rxor__', '__set__', '__setitem__', '__setslice__',
    '__str__', '__sub__', '__truediv__', '__unicode__', '__xor__',
]

# Not included: [
#   '__call__', '__coerce__', '__del__', '__dict__', '__getattr__',
#   '__getattribute__', '__init__', '__new__', '__setattr__'
# ]


def _get_sym_magic_method(name):
    def magic_method(self, *args, **kwargs):
        return Call(GetAttr(self, name), args, kwargs)
    return magic_method


for name in _magic_method_names:
    setattr(Expression, name, _get_sym_magic_method(name))

# subtypes --------------------------------------------------------------------


@dataclass(frozen=True)
class Symbol(Expression, DataFrame):
    _name: str

    def eval_symbolic(self, env):
        return env[self._name]

    def __str__(self):
        return f"{self._name}"


@dataclass(frozen=True)
class GetAttr(Expression):
    _obj: Any
    _name: str

    def eval_symbolic(self, env):
        evaled_obj = eval_symbolic(self._obj, env)
        return getattr(evaled_obj, self._name)

    def __str__(self):
        return f"{self._obj}.{self._name}"


class Call(Expression):
    _func: Callable
    _args: Sequence
    _kwargs: Mapping

    def __init__(self, func, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs
        # print('\n', 'Call.__init__[', self.func, '|',
        #       self._args, '|', self._kwargs, ']')

    def eval_symbolic(self, env):
        # print('Call.eval_symbolic[', self.func, '|', self._args, '|', self._kwargs)
        evaled_func = eval_symbolic(self._func, env)
        evaled_args = []
        if self._args is not None:
            args = [a for tup in self._args for a in tup]
            pargs = self._kwargs.get('args')
            if pargs is not None:
                args += pargs
            evaled_args = [eval_symbolic(v, env) for v in args]

        evaled_kwargs = {}
        if self._kwargs is not None:
            kwargs = self._kwargs.get('kwargs')
            if kwargs is not None and len(kwargs) > 0:
                evaled_kwargs = {k: eval_symbolic(v, env)
                                 for k, v in kwargs.items()}
            # print('Call._evaluated[', evaled_func, '|',
            #       evaled_args, '|', evaled_kwargs, ']')
        return evaled_func(*evaled_args, **evaled_kwargs)

    def __str__(self):
        args = ""
        if self._args is not None and len(self._args) > 0:
            args = ', '.join(str(a) for tup in self._args for a in tup)
        if self._kwargs is not None:
            pargs = self._kwargs.get('args')
            if pargs is not None:
                args += ', ' if len(args) > 0 else ""
                args += ', '.join(str(tup) for tup in pargs)
            kwargs = self._kwargs.get('kwargs')
            if kwargs is not None and len(kwargs) > 0:
                args += ', ' if len(args) > 0 else ""
                args += ', '.join(f'{k}={v}' for k,
                                  v in kwargs.items())
        return f'{self._func}({args})'


def eval_symbolic(obj, env):
    # print('eval_symbolic', obj, env, obj.eval_symbolic(env) if hasattr(obj, 'eval_symbolic') else obj)
    return obj.eval_symbolic(env) if hasattr(obj, 'eval_symbolic') else obj


# The super variable....]
me = Symbol('me')
