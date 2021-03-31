import builtins
import operator
import unittest
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

from torcharrow import Call, Expression, GetAttr, Var, eval_expression

# -----------------------------------------------------------------------------


# All kinds of callig conventions

@ dataclass
class Cell:
    val: int

    def get1(self):
        return self.val

    # positional args
    def get2(self, x):
        return self.val+x

    # varargs
    def get3(self, *xs):
        return self.val+sum(xs)

    # pos + varargs
    def get4(self, x, *xs):
        return self.val+5*x+sum(xs)

    # default
    def get5(self, n=100):
        return self.val+n

    # kwargs
    def get6(self, /, n, m):
        return self.val+n + m

    # pos or kwargs
    def get6(self, n=100, m=200):
        return self.val+n + m

    @ staticmethod
    def get7():
        return 77

    @ classmethod
    def get8(cls, x):
        return x*8

    def map(self, fun):
        return fun(self.val)


def succ(x): return x+1

# Global dunder nethods (like len):


def len(x):
    if isinstance(x, Expression):
        return Call(GetAttr(x, "__len__"))
    else:
        return builtins.len(x)


class TestSymbolicExpression(unittest.TestCase):

    def test_builtins(self):
        me = Var('me')
        ME = Var('ME')

        self.assertEqual(str(me), 'me')
        self.assertEqual(str(me+1), "me.__add__(1)")
        self.assertEqual(str(1+me), "me.__radd__(1)")
        self.assertEqual(str(1+me*me), "me.__mul__(me).__radd__(1)")
        self.assertEqual(str(me.abs()), "me.abs()")

        env = {'me': 1}
        self.assertEqual(eval_expression(1, {'me': 1}), 1)
        self.assertEqual(eval_expression(me + 1, {'me': 1}), 2)
        self.assertEqual(eval_expression(1 + me, {'me': 1}), 2)
        self.assertEqual(eval_expression(1 + me, {'me': 1}), 2)
        self.assertEqual(eval_expression([1, 2, 3], {}), [1, 2, 3])
        self.assertEqual(eval_expression(len([1, 2, 3]), {}), 3)

        self.assertEqual(str(len(me)),  "me.__len__()")

        self.assertEqual(str(me.get1()), "me.get1()")
        self.assertEqual(str(me.get2(2)), "me.get2(2)")
        self.assertEqual(str(me.get3(2, 2, 3, 4)), "me.get3(2, 2, 3, 4)")
        self.assertEqual(str(me.get4(2, 2, 3, 4)), "me.get4(2, 2, 3, 4)")
        self.assertEqual(str(me.get5()), "me.get5()")
        self.assertEqual(str(me.get5(100)), "me.get5(100)")
        self.assertEqual(str(me.get6(n=1, m=2)), "me.get6(n=1, m=2)")
        self.assertEqual(str(me.get6(m=2)), "me.get6(m=2)")
        self.assertEqual(str(ME.get7(m=2)), "ME.get7(m=2)")
        self.assertEqual(str(ME.get8(1000)), "ME.get8(1000)")

        env = {'me': Cell(12), 'ME': Cell}
        self.assertEqual(eval_expression(me.get1(), env), 12)
        self.assertEqual(eval_expression(me.get2(2), env), 14)
        self.assertEqual(eval_expression(me.get3(2, 2, 3, 4), env), 23)
        self.assertEqual(eval_expression(me.get4(2, 2, 3, 4), env), 31)
        self.assertEqual(eval_expression(me.get5(), env), 112)
        self.assertEqual(eval_expression(me.get5(101), env), 113)
        self.assertEqual(eval_expression(me.get6(n=1, m=2), env), 15)
        self.assertEqual(eval_expression(me.get6(m=2), env), 114)
        self.assertEqual(eval_expression(ME.get7(), env), 77)
        self.assertEqual(eval_expression(ME.get8(1000), env), 8000)

        self.assertEqual(eval_expression(me.map(succ), env), 13)


if __name__ == '__main__':
    unittest.main()
