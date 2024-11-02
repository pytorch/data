# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Iterable, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T", covariant=True)


class BaseNodeIterator(Iterator[T]):
    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def started(self) -> bool:
        raise NotImplementedError()

    def finished(self) -> bool:
        raise NotImplementedError()


class BaseNode(Iterable[T]):
    _it: Optional[BaseNodeIterator[T]] = None  # holds pointer to last iter() requested
    _initial_state: Optional[Dict[str, Any]] = None

    def iterator(self, initial_state: Optional[dict]) -> Iterator[T]:
        """Override this method to implement the iterator.
        Iterators are expected to raise StopIteration to signal
        end of iteration, so they can be used in for loops.
        Generators just need to return, as usual.

        initial_state will be passed if `load_state_dict(initial_state)` was called
        on this node before the __iter__ is requested, otherwise None will be passed
        """
        raise NotImplementedError()

    def get_state(self) -> Dict[str, Any]:
        """Return a dictionary that can be passed to iterator(...) which
        can be used to initialize iterator at a certain state.
        """
        raise NotImplementedError()

    def __iter__(self) -> BaseNodeIterator[T]:
        if self._it is not None and self._it.started():
            # Only create a new iter if the last requested one did not start/finish
            self._it = None

        if self._it is None:
            self._it = _EagerIter(self, self._initial_state)
            self._initial_state = None
        return self._it

    def state_dict(self) -> Dict[str, Any]:
        if self._it is None:
            logger.info("state_dict() on BaseNode before __iter__ requested, instantiating iterator to make the call")
            iter(self)
        assert self._it is not None
        return self._it.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._initial_state = state_dict


class _EagerIter(BaseNodeIterator[T]):
    """
    Basic iterator which will runs next-calls eagerly
    """

    STARTED_KEY = "started"
    FINISHED_KEY = "finished"
    BASE_NODE_KEY = "base_node"

    def __init__(self, base_node: BaseNode[T], initial_state: Optional[Dict[str, Any]]):
        self.base_node = base_node
        if initial_state is not None:
            self._it = self.base_node.iterator(initial_state[self.BASE_NODE_KEY])
            self._started = initial_state[self.STARTED_KEY]
            self._finished = initial_state[self.FINISHED_KEY]
        else:
            self._it = self.base_node.iterator(None)
            self._started = False
            self._finished = False

    def __next__(self) -> T:
        self._started = True
        try:
            return next(self._it)
        except StopIteration:
            self._finished = True
            raise

    def state_dict(self) -> Dict[str, Any]:
        return {
            self.BASE_NODE_KEY: self.base_node.get_state(),
            self.STARTED_KEY: self._started,
            self.FINISHED_KEY: self._finished,
        }

    def started(self) -> bool:
        return self._started

    def finished(self) -> bool:
        return self._finished
