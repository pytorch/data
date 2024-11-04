# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, TypeVar

from torchdata.nodes.base_node import BaseNode, T

from .map import Mapper

from .types import Stateful

K = TypeVar("K", covariant=True)


class IterableWrapper(BaseNode[T]):
    """Thin Wrapper that converts any Iterable (including
    torch.utils.data.IterableDataset) in to a BaseNode.

    If iterable implements the Stateful Protocol, it will be saved and restored with its
    state_dict/load_state_dict methods.

    If the iterator resulting from iter(iterable) is Stateful it is IGNORED.

    :param iterable: Iterable to wrap. IterableWrapper calls iter() on it.
    """

    NUM_YIELDED_KEY = "_num_yielded"
    ITERABLE_KEY = "iterable"

    def __init__(self, iterable: Iterable[T]):
        self.iterable = iterable
        self._num_yielded = 0

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        if initial_state is not None:
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
            if isinstance(self.iterable, Stateful):
                self.iterable.load_state_dict(initial_state[self.ITERABLE_KEY])
                it = iter(self.iterable)
            else:
                it = iter(self.iterable)
                # Naively fast-forwarding
                for _ in range(self._num_yielded):
                    next(it)
        else:
            it = iter(self.iterable)

        for item in it:
            self._num_yielded += 1
            yield item

    def get_state(self) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {self.NUM_YIELDED_KEY: self._num_yielded}
        if isinstance(self.iterable, Stateful):
            state_dict[self.ITERABLE_KEY] = self.iterable.state_dict()
        return state_dict


def MapStyleWrapper(map_dataset: Mapping[K, T], sampler: Iterable[K]) -> BaseNode[T]:
    """Thin Wrapper that converts any MapDataset in to a torchdata.node
    If you want parallelism, copy this and replace Mapper with ParallelMapper.

    :param map_dataset: Mapping to wrap.
    :param sampler: Optional[Iterable].
    """
    sampler_node = IterableWrapper(sampler)
    mapper_node = Mapper(sampler_node, map_dataset.__getitem__)
    return mapper_node
