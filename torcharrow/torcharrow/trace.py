import functools
from typing import List, Tuple, Type, Optional

from .expression import Expression, Call, Var, GetAttr

# -----------------------------------------------------------------------------
# Trace state (global)
#
# TODO for parallelization of the FB's test runner
#      make this thread local (assuming the same thread executes the code)


class Trace:

    # class fields
    _trace: List[Tuple[id, Expression]] = []
    _nesting_level = 0
    _is_on = False
    _types = ()

    @classmethod
    def reset(cls):
        cls._trace: List[Tuple[id, Expression]] = []
        cls._nesting_level = 0
        cls._is_on = False
        cls._types = ()

    @classmethod
    def repr(cls):
        return f"Trace:: {cls._is_on}, {cls._types}, {cls._nesting_level}, {'; '.join(f'{id} = {exp}' for id, exp in cls._trace)}"

    @classmethod
    def turn(cls, on: bool, types: Optional[Tuple[Type, ...]] = None):
        cls._is_on = on
        if on:
            cls._trace = []
        if types is not None:
            cls._types = types

    @classmethod
    def is_on(cls):
        return cls._is_on

    @classmethod
    def append(cls, call: Expression):
        cls._trace.append(call)

    @classmethod
    def iter(cls):
        return iter(cls._trace)

    @classmethod
    def statements(cls):
        res = []
        for id, exp in cls._trace:
            # print('STM', 'exp', exp, 'func', exp._func, 'prop', hasattr(
            #     exp._func, "_is_property"), 'call', isinstance(exp, Call))
            # if hasattr(exp._func, "_is_property"):
            #     res.append(f"{id}={exp._args[0]}.{exp._func.__name__}")
            # __nel
            if isinstance(exp, Call) and exp._func.__name__ == "__init__":
                # str(exp) deletes first arg for constructors..
                id = exp._args[0]
                # delete __init__
                res.append(f"{id} = {str(exp).replace('.__init__','')}")

            else:
                res.append(f"{id} = {exp}")
        return res

    @classmethod
    def result(cls):
        for i in range(len(cls._trace)-1, -1, -1):
            id, expr = cls._trace[i]
            if id != "_":
                return id
        return None


# -----------------------------------------------------------------------------
# function decorator


def trace(fn):
    """Trace all basic torcharrow functions operating on given types."""

    @ functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # print("TRACE-IN", fn, "args", args, "kwargs", kwargs)

        # existing functions
        Trace._nesting_level += 1
        res = fn(*args, **kwargs)
        Trace._nesting_level -= 1

        # handle top level primitive functions
        if Trace._nesting_level == 0 and Trace.is_on():
            # print("TRACE-MID", fn, "args", args, "kwargs", kwargs)
            args_ = []
            for a in args:
                if isinstance(a, Trace._types):
                    args_.append(Var(a.id))
                else:
                    args_.append(a)
            kwargs_ = {}
            for k, v in kwargs.items():
                if isinstance(v, Trace._types):
                    kwargs_[k] = Var(v.id)
                else:
                    kwargs_[k] = v
            out = "_"
            if isinstance(res, Trace._types):
                out = res.id
            # print("TRACE-END", fn, "args", args_, "kwargs", kwargs_)
            Trace.append(
                (out, Call(fn, args_, kwargs_)))

        return res

    return wrapped


def traceproperty(fn):
    """Trace all basic torcharrow functions operating on given types."""

    @ functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # same code as above, except for this line...
        fn._is_property = True

        # existing functions
        Trace._nesting_level += 1
        res = fn(*args, **kwargs)
        Trace._nesting_level -= 1

        # handle top level primitive functions
        if Trace._nesting_level == 0 and Trace.is_on():
            # print("TRACE PROP", fn, [str(a) for a in args], kwargs)
            args_ = []
            for a in args:
                if isinstance(a, Trace._types):
                    args_.append(Var(a.id))
                else:
                    args_.append(a)
            # print("TRACE PROP", fn, [str(a) for a in args_], kwargs)
            kwargs_ = {}
            for k, v in kwargs.items():
                if isinstance(a, Trace._types):
                    kwargs_[k] = args_.Var(v.id)
                else:
                    kwargs_[k] = v
            out = "_"
            if isinstance(res, Trace._types):
                out = res.id
            Trace.append(
                (out, Call(fn, tuple(args_), kwargs_)))

        return res

    return wrapped
