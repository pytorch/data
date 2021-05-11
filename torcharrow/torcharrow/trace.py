import functools
from typing import List, Tuple, Type, Optional

from .expression import Expression, Call, Var, GetAttr

# -----------------------------------------------------------------------------
# Trace state (part of a scope object)


class Trace:
    # singleton, only instantiated in

    def __init__(self, is_on, types_to_trace):
        self._trace: List[Tuple[str, Expression]] = []
        self._nesting_level = 0
        self._is_on = is_on
        self._types: Tuple[Type, ...] = types_to_trace

    def reset(self):
        self._trace: List[Tuple[str, Expression]] = []
        self._nesting_level = 0
        self._is_on = False
        self._types = ()

    def repr(self):
        return f"Trace:: {self._is_on}, {self._types}, {self._nesting_level}, {'; '.join(f'{id} = {exp}' for id, exp in self._trace)}"

    # def turn(self, on: bool, types: Optional[Tuple[Type, ...]] = None):
    #     self._is_on = on
    #     if on:
    #         self._trace = []
    #     if types is not None:
    #         self._types = types

    def is_on(self):
        return self._is_on

    def append(self, call: Tuple[str, Expression]):
        self._trace.append(call)

    def iter(self):
        return iter(self._trace)

    def statements(self):
        res = []
        for id, exp in self._trace:
            # print('STM', 'exp', exp, 'func', exp._func, 'prop', hasattr(
            #     exp._func, "_is_property"), 'call', isinstance(exp, Call))
            # if hasattr(exp._func, "_is_property"):
            #     res.append(f"{id}={exp._args[0]}.{exp._func.__name__}")
            # __nel
            if (
                isinstance(exp, Call)
                and exp._func.__name__ == "__init__"
                and not id.startswith("s")
            ):
                # str(exp) deletes first arg for constructors..
                id = exp._args[0]
                # delete __init__
                res.append(f"{id} = {str(exp).replace('.__init__','')}")

            else:
                res.append(f"{id} = {exp}")
        return res

    def result(self):
        for i in range(len(self._trace) - 1, -1, -1):
            id, expr = self._trace[i]
            if id != "_":
                return id
        return None


# -----------------------------------------------------------------------------
# function decorator


def get_trace(*args, **kwargs):
    from .scope import Scope

    for arg in args:
        if isinstance(arg, Scope):
            return arg.trace
        if hasattr(arg, "_scope") and isinstance(arg._scope, Scope):
            return arg._scope.trace
        elif "scope" in kwargs and isinstance(kwargs["scope"], Scope):
            return kwargs["scope"].trace
    return None


def trace(fn):
    """Trace all basic torcharrow functions operating on given types."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        trace = get_trace(*args, **kwargs)
        # if trace is None:
        #     raise TypeError('function to trace must have a scope argument')

        # # print("TRACE-IN", fn, "args", args, "kwargs", kwargs)

        # # existing functions
        trace._nesting_level += 1
        res = fn(*args, **kwargs)
        trace._nesting_level -= 1

        # # handle top level primitive functions
        if trace._nesting_level == 0 and trace.is_on():
            # print("TRACE-MID", fn, "args", args, "kwargs", kwargs)
            args_ = []
            for a in args:
                if isinstance(a, trace._types):
                    args_.append(Var(a.id))
                else:
                    args_.append(a)
            kwargs_ = {}
            for k, v in kwargs.items():
                if isinstance(v, trace._types):
                    kwargs_[k] = Var(v.id)
                else:
                    kwargs_[k] = v
            out = "_"
            if isinstance(res, trace._types):
                out = res.id
            # print("TRACE-END", fn, "args", args_, "kwargs", kwargs_)
            trace.append((out, Call(fn, args_, kwargs_)))

        return res

    return wrapped


def traceproperty(fn):
    """Trace all basic torcharrow functions operating on given types."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # # same code as above, except for this line...
        # fn._is_property = True
        # #find the scope::
        # # at least one positional argument must be an IColumn
        # for arg in args:
        #     if isinstance(arg,IColumn):
        #         scope = arg.scope

        # # existing functions
        # trace._nesting_level += 1
        res = fn(*args, **kwargs)
        # trace._nesting_level -= 1

        # # handle top level primitive functions
        # if trace._nesting_level == 0 and trace.is_on():
        #     # print("TRACE PROP", fn, [str(a) for a in args], kwargs)
        #     args_ = []
        #     for a in args:
        #         if isinstance(a, trace._types):
        #             args_.append(Var(a.id))
        #         else:
        #             args_.append(a)
        #     # print("TRACE PROP", fn, [str(a) for a in args_], kwargs)
        #     kwargs_ = {}
        #     for k, v in kwargs.items():
        #         if isinstance(a, trace._types):
        #             kwargs_[k] = args_.Var(v.id)
        #         else:
        #             kwargs_[k] = v
        #     out = "_"
        #     if isinstance(res, trace._types):
        #         out = res.id
        #     trace.append((out, Call(fn, tuple(args_), kwargs_)))

        return res

    return wrapped
