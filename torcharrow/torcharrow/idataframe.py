#!/usr/bin/env python3
from __future__ import annotations
import abc

import array as ar
import functools
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

import torcharrow.dtypes as dt

from .column_factory import ColumnFactory, Device
from .icolumn import IColumn
from .expression import Var, eval_expression, expression
from .scope import Scope
from .trace import trace, traceproperty

# assumes that these have been imported already:
# from .inumerical_column import INumericalColumn
# from .istring_column import IStringColumn
# from .imap_column import IMapColumn
# from .ilist_column import IMapColumn

# ------------------------------------------------------------------------------
# DataFrame Factory with default scope and device


def DataFrame(
    data: Union[Iterable, dt.DType, Literal[None]] = None,
    dtype: Optional[dt.DType] = None,
    scope: Optional[Scope] = None,
    to: Device = "",
):
    scope = scope or Scope.default
    to = to or scope.to
    return scope.Frame(data, dtype=dtype, to=to)


# -----------------------------------------------------------------------------
# DataFrames aka (StructColumns, can be nested as StructColumns:-)

DataOrDTypeOrNone = Union[Mapping, Sequence, dt.DType, Literal[None]]


class IDataFrame(IColumn):
    """Dataframe, ordered dict of typed columns of the same length"""

    def __init__(self, scope, to, dtype):
        assert dt.is_struct(dtype)
        super().__init__(scope, to, dtype)

    @property  # type: ignore
    @abc.abstractmethod
    def columns(self):
        """The column labels of the DataFrame."""
        return [f.name for f in self.dtype.fields]


# TODO Make this abstract and add all the abstract methods here ...
# TODO Current short cut has 'everything', excpet for columns as a  DataFrameStd
# TODO Make GroupedDatFrame also an IGroupedDataframe to make it truly compositional


# -----------------------------------------------------------------------------
# DataFrameVariable me


class IDataFrameVar(Var, IDataFrame):
    # A dataframe variable is purely symbolic,
    # It should only appear as part of a relational expression

    def __init__(self, name: str, qualname: str = ""):
        super().__init__(name, qualname)

    def _append_null(self):
        return self._not_supported("_append_null")

    def _append_value(self, value):
        return self._not_supported("_append_value")

    def _append_data(self, value):
        return self._not_supported("_append_data")

    def _finalize(self, mask=None):
        return self._not_supported("_finalize")

    def __len__(self):
        return self._not_supported("len")

    def null_count(self):
        return self._not_supported("null_count")

    def getmask(self, i):
        return self._not_supported("getmask")

    def getdata(self, i):
        return self._not_supported("getdata")

    @property  # type: ignore
    def columns(self):
        return self._not_supported("getdata")


# The super variable...
me = IDataFrameVar("me", "torcharrow.idataframe.me")
