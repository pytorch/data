# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Generic, Iterable, Iterator, Mapping, Optional, Sized, TypeVar

from torch.utils.data import IterableDataset, Sampler, SequentialSampler

from torchdata.nodes.base_node import BaseNode, T

K = TypeVar("K", covariant=True)


class IterableWrapper(BaseNode[T]):
    """Thin Wrapper that converts any Iterable (including
    torch.utils.data.IterableDataset) in to a BaseNode.

    :param iterable: Iterable to wrap. IterableWrapper calls iter() on it.
    """

    iterable: Iterable[T]

    def __init__(self, iterable: Iterable[T]):
        self.iterable = iterable

    def iterator(self) -> Iterator[T]:
        return iter(self.iterable)


class MapStyleWrapper(BaseNode[T], Generic[K, T]):
    """Thin Wrapper that converts any Mapping[K, T] into a BaseNode[T].
    If no sampler is provided, a SequentialSampler is used and requires dataset to be Sized.

    Note that if your map_style lookup is expensive, you might want
    to use __to_be_named_dataloader_drop_in__ instead which can take advantage
    of process- or thread-based parallelism.
    """

    dataset: Mapping[K, T]
    sampler: Sampler[K]

    def __init__(self, dataset: Mapping[K, T], sampler: Optional[Sampler[K]] = None):
        self.dataset = dataset
        if sampler is None:
            if not isinstance(self.dataset, Sized):
                raise ValueError("If dataset does not implement __len__, you must pass a sampler!")
            self.sampler = SequentialSampler(self.dataset)  # type: ignore
        else:
            self.sampler = sampler

    def iterator(self) -> Iterator[T]:
        for key in self.sampler:
            yield self.dataset[key]


class ToIterableDataset(IterableDataset[T]):
    def __init__(self, base_node: BaseNode[T]):
        self.base_node = base_node

    def __iter__(self) -> Iterator[T]:
        return iter(self.base_node)
