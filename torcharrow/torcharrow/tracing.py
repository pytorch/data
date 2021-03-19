import sys


def format_arg_value(arg_val):
    """Return a string representing a (name, value) pair.

    >>> format_arg_value(('x', (1, 2, 3)))
    'x=(1, 2, 3)'
    """
    arg, val = arg_val
    return "%s=%r" % (arg, val)


def echo(fn, file=sys.stdout):
    """Echo calls to a function.

    Returns a decorated version of the input function which "echoes" calls
    made to it by writing out the function's name and the arguments it was
    called with.
    """
    import functools

    # Unpack function's arg count, arg names, arg defaults
    print(">>>>", fn, "\n", str(type(fn)))

    code = fn.__code__

    argcount = code.co_argcount
    argnames = code.co_varnames[:argcount]
    fn_defaults = fn.__defaults__ or list()
    argdefs = dict(zip(argnames[-len(fn_defaults) :], fn_defaults))

    @functools.wraps(fn)
    def wrapped(*v, **k):
        # Collect function arguments by chaining together positional,
        # defaulted, extra positional and keyword arguments.
        positional = list(map(format_arg_value, zip(argnames, v)))
        defaulted = [
            format_arg_value((a, argdefs[a])) for a in argnames[len(v) :] if a not in k
        ]
        nameless = list(map(repr, v[argcount:]))
        keyword = list(map(format_arg_value, k.items()))
        args = positional + defaulted + nameless + keyword
        print(f"{fn.__name__}({', '.join(args)})\n")
        # start of checking

        # existing functions
        return fn(*v, **k)

    return wrapped


@echo
def f(x):
    pass


@echo
def g(x, y):
    pass


@echo
def h(x=1, y=2):
    pass


@echo
def i(x, y, *v):
    pass


@echo
def j(x, y, *v, **k):
    pass


class X(object):
    @echo
    def __init__(self, help):
        pass

    @echo
    def m(self, x):
        pass

    @classmethod
    @echo
    def cm(klass, x):
        pass

    @staticmethod
    @echo
    def sm(x):
        pass


# def reversed_write(s):
#     sys.write(''.join(reversed(s)))
# def k(**kw):
#     pass

# k = echo(k, write=reversed_write)


if __name__ == "__main__":
    f(10)
    g("spam", 42)
    g(y="spam", x=42)
    h()
    i("spam", 42, "extra", "args", 1, 2, 3)
    j(("green", "eggs"), y="spam", z=42)
    X("a").m("method call")
    X.cm("classmethod call")
    h = X.cm

    h("classmethod call")

    X.sm("bar")

    X(14)

    # k(born="Mon", christened="Tues", married="Weds")

from typing import TypedDict


class Point(TypedDict):
    x: int
    y: int


# Error: Incompatible types (expression has type "float",
#        TypedDict item "x" has type "int")  [typeddict-item]
p: Point = {"x": 1.2, "y": 4}
