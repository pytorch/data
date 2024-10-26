# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Iterable, Iterator, Mapping, Optional, Sized, TypeVar

from torch.utils.data import Sampler, SequentialSampler

from torchdata.nodes.base_node import BaseNode, T

K = TypeVar("K")


class IterableWrapper(BaseNode[T]):
    """Thin Wrapper that converts any Iterable (including
    torch.utils.data.IterableDataset) in to a BaseNode.

    :param iterable: Iterable to wrap. IterableWrapper calls iter() on it.
    """

    def __init__(self, iterable: Iterable[T]):
        self.iterable = iterable

    def iterator(self) -> Iterator[T]:
        return iter(self.iterable)


class MapStyleWrapper(BaseNode[T]):
    """Thin Wrapper that converts any Mapping[K, T] into a BaseNode[T].
    If no sampler is provided, a SequentialSampler is used and requires dataset to be Sized.

    Note that if your map_style lookup is expensive, you might want
    to use __to_be_named_dataloader_drop_in__ instead which can take advantage
    of process- or thread-based parallelism.
    """

    def __init__(self, dataset: Mapping[K, T], sampler: Optional[Sampler[K]] = None):
        self.dataset = dataset
        if sampler is None:
            if not isinstance(self.dataset, Sized):
                raise ValueError("If dataset does not implement __len__, you must pass a sampler!")
            sampler = SequentialSampler(self.dataset)

        self.sampler: Sampler[K] = sampler

    def iterator(self) -> Iterator[T]:
        for key in self.sampler:
            yield self.dataset[key]
