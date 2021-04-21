import torch
from dataclasses import dataclass
from collections import OrderedDict
from typing import TypeVar, Generic, Union, List, Any, Optional
from .dtypes import DType, is_numerical, is_struct, is_list, is_map, is_string
from .column import Column
from .dataframe import DataFrame
from . import dtypes

T = TypeVar("T")
KT = TypeVar("KT")


@dataclass
class WithPresence(Generic[T]):
    values: T
    # 1-D bool vector of per-element validity bit
    presence: torch.Tensor


@dataclass
class PackedList(Generic[T]):
    # 1-D int32 tensor
    offsets: torch.Tensor
    # Type hints don't really work because of recursive type remapping
    values: Union[List[T], torch.Tensor, Any]


@dataclass
class PackedMap(Generic[KT, T]):
    # 1-D int32 tensor
    offsets: torch.Tensor
    keys: Union[List[KT], torch.Tensor, Any]
    values: Union[List[T], torch.Tensor, Any]


def infer_dtype_from_torch(data):
    raise NotImplementedError()


def from_torch(data: Any, dtype: Optional[DType] = None):
    if dtype is None:
        dtype = infer_dtype_from_torch(data)
    assert isinstance(dtype, DType)

    if isinstance(data, list):
        # if it's a python list - we're switching to python representation for this subtree. This path is also taken for strings, because they are represented as List[str] in PyTorch
        return Column(data, dtype=dtype)

    # handling nullability
    if isinstance(data, WithPresence):
        if not dtype.nullable:
            raise ValueError(
                f"Expected nullable type when the value is pytorch.WithPresence: {dtype}"
            )
        nested = from_torch(data.values, dtype=dtype.with_null(False))
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        return Column(
            [(x if data.presence[i].item() else None) for i, x in enumerate(nested)],
            dtype=dtype,
        )
    if dtype.nullable:
        raise ValueError(
            f"nullable dtype must be represented by pytorch.WithPresence: {dtype}"
        )

    # container types
    if isinstance(data, PackedList):
        if not is_list(dtype):
            raise ValueError(
                f"Expected list type when the value is pytorch.PackedList: {dtype}"
            )
        assert isinstance(dtype, dtypes.List_)  # make mypy happy
        nested = list(from_torch(data.values, dtype=dtype.item_dtype))
        if not isinstance(data.offsets, torch.Tensor) or data.offsets.dtype not in [
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            raise ValueError(
                "PackedList.offsets is expected to be an integer-valued tensor"
            )
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        offsets = data.offsets.tolist()
        return Column(
            [nested[offsets[i] : offsets[i + 1]] for i in range(len(data.offsets) - 1)],
            dtype=dtype,
        )
    if isinstance(data, PackedMap):
        if not is_map(dtype):
            raise ValueError(
                f"Expected map type when the value is pytorch.PackedMap: {dtype}"
            )
        assert isinstance(dtype, dtypes.Map)  # make mypy happy
        nested_keys = list(from_torch(data.keys, dtype=dtype.key_dtype))
        nested_values = list(from_torch(data.values, dtype=dtype.item_dtype))
        if not isinstance(data.offsets, torch.Tensor) or data.offsets.dtype not in [
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            raise ValueError(
                "PackedMap.offsets is expected to be an integer-valued tensor"
            )
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        offsets = data.offsets.tolist()
        return Column(
            [
                OrderedDict(
                    zip(
                        nested_keys[offsets[i] : offsets[i + 1]],
                        nested_values[offsets[i] : offsets[i + 1]],
                    )
                )
                for i in range(len(data.offsets) - 1)
            ],
            dtype=dtype,
        )
    if isinstance(data, tuple):
        # TODO: check that fields of named tuples match?
        if not is_struct(dtype):
            raise ValueError(
                f"Expected map type when the value is pytorch.PackedMap: {dtype}"
            )
        assert isinstance(dtype, dtypes.Struct)  # make mypy happy
        if len(data) != len(dtype.fields):
            raise ValueError(
                f"Tuple has {len(data)} fields when the type has {len(dtype.fields)}: {dtype}"
            )
        nested_fields = OrderedDict(
            (
                dtype.fields[i].name,
                list(from_torch(data[i], dtype=dtype.fields[i].dtype)),
            )
            for i in range(len(data))
        )
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        return DataFrame(nested_fields, dtype=dtype)

    # numerics!
    if isinstance(data, torch.Tensor):
        # lazy way of converting types
        torcharrow_dtype_name = str(data.dtype)
        assert torcharrow_dtype_name.startswith("torch.")
        torcharrow_dtype_name = torcharrow_dtype_name[len("torch.") :]
        if torcharrow_dtype_name == "bool":
            torcharrow_dtype_name = "boolean"
        if not hasattr(dtypes, torcharrow_dtype_name):
            raise ValueError(f"Unexpected dtype for the tensor: {data.dtype}")
        if getattr(dtypes, torcharrow_dtype_name) != dtype:
            raise ValueError(
                f"Unexpected dtype {data.dtype} for the tensor, expected {dtype}"
            )
        # TODO: this implementation is very inefficient, we should wrap the column directly instead of round-tripping through python
        return Column(data.tolist(), dtype=dtype)

    raise ValueError(f"Unexpected data in `from_torch`: {type(data)}")
