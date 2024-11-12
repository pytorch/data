# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, TypeVar

from torch.utils.data import Sampler

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
        self._it = None

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        # if self._it is not None and self._it._started:
        #     self._it = None
        self._it = self.Iter(self, initial_state)
        return self._it

    def get_state(self) -> Dict[str, Any]:
        if self._it is None:
            iter(self)
        assert self._it is not None
        return self._it.get_state()

    class Iter(Iterator[T]):
        def __init__(self, parent, initial_state: Optional[Dict[str, Any]]):
            self.parent = parent
            self._num_yielded = 0
            if initial_state is not None:
                self._num_yielded = initial_state[parent.NUM_YIELDED_KEY]
                if isinstance(parent.iterable, Stateful):
                    parent.iterable.load_state_dict(initial_state[parent.ITERABLE_KEY])
                    self._it = iter(parent.iterable)
                else:
                    self._it = iter(parent.iterable)
                    # Naively fast-forwarding
                    for i in range(self._num_yielded):
                        try:
                            next(self._it)
                        except StopIteration:
                            raise ValueError(
                                f"Tried to fast-forward {self._num_yielded} items during init but "
                                f"hit StopIteration after {i} items, this is likely a bug or malformed state_dict"
                            )
            else:
                self._it = iter(parent.iterable)

        def __iter__(self):
            return self

        def __next__(self):
            item = next(self._it)
            self._num_yielded += 1
            return item

        def get_state(self) -> Dict[str, Any]:
            state_dict: Dict[str, Any] = {self.parent.NUM_YIELDED_KEY: self._num_yielded}
            if isinstance(self.parent.iterable, Stateful):
                state_dict[self.parent.ITERABLE_KEY] = self.parent.iterable.state_dict()
            return state_dict


def MapStyleWrapper(map_dataset: Mapping[K, T], sampler: Sampler[K]) -> BaseNode[T]:
    """Thin Wrapper that converts any MapDataset in to a torchdata.node
    If you want parallelism, copy this and replace Mapper with ParallelMapper.

    :param map_dataset: Mapping[K, T] - Apply map_dataset.__getitem__ to the outputs of sampler.
    :param sampler: Sampler[K]
    """
    sampler_node: SamplerWrapper[K] = SamplerWrapper(sampler)
    mapper_node = Mapper(sampler_node, map_dataset.__getitem__)
    return mapper_node


class SamplerWrapper(BaseNode[T]):
    """
    Convert a sampler into a BaseNode. This is nearly identical to
    IterableWrapper except it includes a hook to call set_epoch on the sampler,
    if it supports it.

    :param sampler: Sampler - to wrap.
    :param initial_epoch: int - initial epoch to set on the sampler
    :param epoch_updater: Optional[Callable[[int], int]] = None - callback to update epoch at start of new iteration. It's called at the beginning of each iterator request, except the first one.
    """

    NUM_YIELDED_KEY = "_num_yielded"
    SAMPLER_KEY = "sampler"
    EPOCH_KEY = "_epoch"
    STARTED_KEY = "_started"

    class Iter(Iterator[T]):
        def __init__(self, parent, initial_state: Optional[Dict[str, Any]]):
            self.parent = parent
            self._num_yielded = 0
            self._epoch = 0
            self._started = False
            if initial_state is not None:
                self._num_yielded = initial_state[parent.NUM_YIELDED_KEY]
                self._epoch = initial_state[parent.EPOCH_KEY]
                self._started = initial_state[parent.STARTED_KEY]

                if isinstance(parent.sampler, Stateful):
                    parent.sampler.load_state_dict(initial_state[parent.SAMPLER_KEY])
                    self._it = iter(parent.sampler)
                else:
                    if hasattr(parent.sampler, "set_epoch"):
                        parent.sampler.set_epoch(self._epoch)
                    self._it = iter(parent.sampler)
                    for i in range(self._num_yielded):
                        try:
                            next(self._it)
                        except StopIteration:
                            raise ValueError(
                                f"Tried to fast-forward {self._num_yielded} items during init but "
                                f"hit StopIteration after {i} items, this is likely a bug or malformed state_dict"
                            )
            else:
                if self._started:  # don't call first time
                    self._epoch = parent.epoch_updater(self._epoch)
                if hasattr(parent.sampler, "set_epoch"):
                    parent.sampler.set_epoch(self._epoch)
                self._it = iter(parent.sampler)

        def __iter__(self):
            return self

        def __next__(self) -> T:
            self._started = True

            item = next(self._it)
            self._num_yielded += 1
            return item

        def get_state(self):
            print("Getting state", self._num_yielded)
            state_dict: Dict[str, Any] = {
                self.parent.NUM_YIELDED_KEY: self._num_yielded,
                self.parent.EPOCH_KEY: self._epoch,
                self.parent.STARTED_KEY: self._started,
            }
            if isinstance(self.parent.sampler, Stateful):
                state_dict[self.parent.SAMPLER_KEY] = self.parent.sampler.state_dict()
            return state_dict

        # for item in it:
        #     self._num_yielded += 1
        #     yield item

    @classmethod
    def _default_epoch_updater(cls, epoch: int) -> int:
        return epoch + 1

    def __init__(
        self,
        sampler: Sampler[T],
        initial_epoch: int = 0,
        epoch_updater: Optional[Callable[[int], int]] = None,
    ):
        self.sampler = sampler
        self.epoch_updater = epoch_updater or self._default_epoch_updater
        self._it = None
        # self._num_yielded = 0
        # self._epoch = initial_epoch
        # self._started = False

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        # if self._it is not None and self._it._started:
        #     self._it = None
        self._it = self.Iter(self, initial_state)
        return self._it
        # it: Iterator[T]
        # self._num_yielded = 0
        # if initial_state is not None:
        #     self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
        #     self._epoch = initial_state[self.EPOCH_KEY]
        #     self._started = initial_state[self.STARTED_KEY]

        #     if isinstance(self.sampler, Stateful):
        #         self.sampler.load_state_dict(initial_state[self.SAMPLER_KEY])
        #         it = iter(self.sampler)
        #     else:
        #         if hasattr(self.sampler, "set_epoch"):
        #             self.sampler.set_epoch(self._epoch)
        #         it = iter(self.sampler)
        #         for i in range(self._num_yielded):
        #             try:
        #                 next(it)
        #             except StopIteration:
        #                 raise ValueError(
        #                     f"Tried to fast-forward {self._num_yielded} items during init but "
        #                     f"hit StopIteration after {i} items, this is likely a bug or malformed state_dict"
        #                 )
        # else:
        #     if self._started:  # don't call first time
        #         self._epoch = self.epoch_updater(self._epoch)
        #     if hasattr(self.sampler, "set_epoch"):
        #         self.sampler.set_epoch(self._epoch)
        #     it = iter(self.sampler)

        # self._started = True
        # for item in it:
        #     self._num_yielded += 1
        #     yield item

    def get_state(self) -> Dict[str, Any]:
        if self._it is None:
            iter(self)
        assert self._it is not None
        return self._it.get_state()

    #     state_dict: Dict[str, Any] = {
    #         self.NUM_YIELDED_KEY: self._num_yielded,
    #         self.EPOCH_KEY: self._epoch,
    #         self.STARTED_KEY: self._started,
    #     }
    #     if isinstance(self.sampler, Stateful):
    #         state_dict[self.SAMPLER_KEY] = self.sampler.state_dict()
    #     return state_dict
