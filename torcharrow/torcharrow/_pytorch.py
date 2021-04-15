import torch
from dataclasses import dataclass
from typing import TypeVar, Generic, Union, List, Any

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
