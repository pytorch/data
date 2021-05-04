import json
from typing import Literal, Optional, Sequence, TypedDict, Union, Iterable, Tuple, Dict, cast, Mapping
from dataclasses import dataclass

# from .trace import Trace
from .config import Config
from .column_factory import ColumnFactory, Device, Typecode

from .dtypes import DType, infer_dtype_from_prefix, is_tuple, Struct, is_struct, Tuple_, Field

from .expression import Call, Var, GetAttr
from .trace import Trace, trace

# ---------------------------------------------------------------------------
# Session, pipelines global state...

# helper class


class Counter():
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    def next(self):
        n = self._value
        self._value += 1
        return n


class Session():

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default
        self.ct = Counter()
        self.id = 's0'
        self._session = self
        self.trace = Trace(self.config['tracing'], tuple(
            self.config['types_to_trace']))
        # if self.trace.is_on():
        #    self.trace.append(('s0', Call(Session.__init__, [self, config], {})))

    # device handling --------------------------------------------------------

    @property
    def to(self):
        return self.config["device"]

    # tracing handling --------------------------------------------------------
    @property
    def tracing(self):
        return self.config["tracing"]

    # only one session --------------------------------------------------------

    def is_same(self, other):
        return id(self) == id(other)

    def check_is_same(self, other):
        if id(self) != id(other) or self.to != other.to:
            raise TypeError('session and device must be the same')

    def check_are_same(self, others):
        if not all(self.is_same(other) and self.to == other.to for other in others):
            raise TypeError('session and device must be the same')

    # column factory -----------------------------------------------------------

    @staticmethod
    def _require_column_constructors_to_be_registered():
        # requires that all columns have registered their factory methods...
        # handles cyclic references...
        from .numerical_column import NumericalColumn
        from .velox_rt.numerical_column_cpu import NumericalColumnCpu
        from .test_rt.numerical_column_test import NumericalColumnTest
        from .string_column import StringColumn
        from .list_column import ListColumn
        from .map_column import MapColumn
        from .dataframe import DataFrame

    # private column/dataframe constructors

    def _Empty(self, dtype, to='', mask=None):
        """
        Column row builder method -- lifecycle:: _Empty; _append_row*; _finalize
        """
        Session._require_column_constructors_to_be_registered()

        to = to if to != '' else self.to
        call = ColumnFactory.lookup((dtype.typecode+"_empty", to))

        return call(self, to, dtype, mask)

    def _Full(self, data, dtype, to='', mask=None):
        """
        Column vector builder method -- data is already in right form
        """
        Session._require_column_constructors_to_be_registered()

        to = to if to != '' else self.to
        call = ColumnFactory.lookup((dtype.typecode+"_full", to))

        return call(self, to, data, dtype, mask)

    # public column (dataframe) constructors
    @trace
    def Column(
        self,
        data: Union[Iterable, DType, Literal[None]] = None,
        dtype: Optional[DType] = None,
        to: Device = ''
    ):
        """
        Column factory method
        """

        to = self.to if to is None else to

        if data is None and dtype is None:
            raise TypeError("Column requires data and/or dtype parameter")

        if isinstance(data, DType) and isinstance(dtype, DType):
            raise TypeError("Column can only have one dtype parameter")

        if isinstance(data, DType):
            (data, dtype) = (dtype, data)

        # dtype given, optional data
        if isinstance(dtype, DType):
            col = self._Empty(dtype, to)
            if data is not None:
                for i in data:
                    col._append(i)
            return col._finalize()

        # data given, optional column
        if data is not None:
            if isinstance(data, Sequence):
                data = iter(data)
            if isinstance(data, Iterable):
                prefix = []
                for i, v in enumerate(data):
                    prefix.append(v)
                    if i > 5:
                        break
                dtype = infer_dtype_from_prefix(prefix)
                if dtype is None:
                    raise TypeError("Column cannot infer type from data")
                if is_tuple(dtype):
                    # TODO fox me
                    raise TypeError(
                        "Column cannot be used to created structs, use Dataframe constructor instead"
                    )
                col = self._Empty(dtype, to=to)
                # add prefix and ...
                for p in prefix:
                    col._append(p)
                # ... continue enumerate the data
                for _, v in enumerate(data):
                    col._append(v)
                return col._finalize()
            else:
                raise TypeError(
                    f"data parameter of Sequence type expected (got {type(dtype).__name__})"
                )
        else:
            raise AssertionError("unexpected case")

    # public dataframe (column)) constructor
    @trace
    def DataFrame(
        self,
        data=None,  # : DataOrDTypeOrNone = None,
        dtype=None,  # : Optional[DType] = None,
        columns=None,  # : Optional[List[str]] = None,
        to=''
    ):
        """
        Dataframe factory method
        """

        if data is None and dtype is None:
            assert columns is None
            return self._Empty(Struct([]), to=to)._finalize()

        if data is not None and isinstance(data, DType):
            if dtype is not None and isinstance(dtype, DType):
                raise TypeError("Dataframe can only have one dtype parameter")
            dtype = data
            data = None

        # dtype given, optional data
        if dtype is not None:
            if not is_struct(dtype):
                raise TypeError(
                    f"Dataframe takes a Struct dtype as parameter (got {dtype})"
                )
            dtype = cast(Struct, dtype)
            if data is None:
                return self._Empty(dtype, to=to)._finalize()
            else:
                if isinstance(data, Sequence):
                    res = self._Empty(dtype, to=to)
                    for i in data:
                        res._append(i)
                    return res._finalize()
                elif isinstance(data, Mapping):
                    res = {n: c if Session._is_column(c) else self.Column(
                        c, to=to) for n, c in data.items()}
                    return self._Full(res, dtype)

                else:
                    raise TypeError(
                        f"Dataframe does not support constructor for data of type {type(data).__name__}"
                    )

        # data given, optional column
        if data is not None:
            if isinstance(data, Sequence):
                prefix = []
                for i, v in enumerate(data):
                    prefix.append(v)
                    if i > 5:
                        break
                dtype = infer_dtype_from_prefix(prefix)
                if dtype is None or not is_tuple(dtype):
                    raise TypeError(
                        "Dataframe cannot infer struct type from data")
                dtype = cast(Tuple_, dtype)
                columns = [] if columns is None else columns
                if len(dtype.fields) != len(columns):
                    raise TypeError(
                        "Dataframe column length must equal row length")
                dtype = Struct(
                    [Field(n, t) for n, t in zip(columns, dtype.fields)]
                )
                res = self._Empty(dtype, to=to)
                for i in data:
                    res._append(i)
                return res._finalize()
            elif isinstance(data, Mapping):
                res = {}
                for n, c in data.items():
                    if Session._is_column(c):
                        res[n] = c
                    elif isinstance(c, Sequence):
                        res[n] = self.Column(c, to=to)
                    else:
                        raise TypeError(
                            f"dataframe does not support constructor for column data of type {type(c).__name__}"
                        )
                return self._Full(res, dtype=Struct([Field(n, c.dtype) for n, c in res.items()]))
            elif Session.is_dataframe(data):
                return data
            else:
                raise TypeError(
                    f"dataframe does not support constructor for data of type {type(data).__name__}"
                )
        else:
            raise AssertionError('unexpected case')

    Frame = DataFrame

    # helper ------------------------------------------------------------------
    @staticmethod
    def _is_column(c):
        # NOTE: shoud be isinstance(c, AbstractColumn)
        # But can't do tha due to cyclic reference, so we use ...
        return hasattr(c, "_dtype") and hasattr(c, "_session") and hasattr(c, "_to")

    @staticmethod
    def _is_dataframe(c):
        # NOTE: shoud be isinstance(c, DataFrame)
        # But can't do tha due to cyclic reference, so we use ...
        return hasattr(c, "_dtype") and hasattr(c, "_session") and hasattr(c, "_to") and hasattr(c, "_field_data")

    def arrange(
        self,
        start: int,
        stop: int,
        step: int = 1,
        dtype: Optional[DType] = None,
        to: Optional[Device] = None
    ):
        return self.Column(list(range(start, stop, step)), dtype, to)


# ------------------------------------------------------------------------------
# registering the default session
Session.default = Session()
