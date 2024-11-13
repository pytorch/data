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
    __it: Optional[BaseNodeIterator[T]] = None  # holds pointer to last iter() requested
    __initial_state: Optional[Dict[str, Any]] = None
    __restart_on_stop_iteration: bool = False

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
        if self.__it is not None and not self.__it.started():
            # Only create a new iter if the last requested one has already started
            return self.__it

        if self.__initial_state is not None:
            self.__it = _EagerIter(self, self.__initial_state)
            self.__initial_state = None
            if self.__restart_on_stop_iteration and not self.__it.has_next():
                self.__it = _EagerIter(self, self.__initial_state)
            self.__restart_on_stop_iteration = False  # reset this for subsequent calls
        else:
            self.__it = _EagerIter(self, self.__initial_state)
        return self.__it

    def state_dict(self) -> Dict[str, Any]:
        return self.get_state()

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        restart_on_stop_iteration: bool = False,
    ) -> None:
        """
        When iter() is next requested from this node, it will be instantiated with state_dict.
        state_dict will be passed directly to the .iterator(...) method of this node which will
        handle proper initialization.

        NOTE [ state_dict end of iteration handling ]
        Special care must be taken when state_dict is requested after StopIteration.
        Consider the following common example of saving state_dict after one epoch.

        ```python
        node: BaseNode = ...
        for batch in node:
            # do something

        state_dict = node.state_dict()
        ```

        Technically, since state_dict() was called before a new iterator was requested from `node`,
        you should expect the following behaviour:

        ```python
        node: BaseNode = ...
        node.load_state_dict(state_dict)  # Load state_dict from above
        next(iter(node))  # Throws StopIteration immediately
        ```

        You can avoid the above (default) behaviour by passing `restart_on_stop_iteration=True` when
        calling `load_state_dict`, eg

        ```python
        node: BaseNode = ...
        node.load_state_dict(state_dict, restart_on_stop_iteration=True)
        next(iter(node))  # Catches StopIteration, creates a new iterator, and returns next()
        ```

        Note: we can not make `True` the default for restart_on_stop_iteration because it would
        prevent StopIteration thrown in leaves from propogating up to the node where load_state_dict is called.
        Instead, wrap the root of your dag in `torchdata.nodes.Loader` which will set this flag for you.

        :param state_dict: state_dict to load in next __iter__ requested
        :param restart_on_stop_iteration: (default False) - whether to restart the iterator automatically
            when the first next() call would throw StopIteration.
        """
        self.__initial_state = state_dict
        self.__restart_on_stop_iteration = restart_on_stop_iteration


class _EagerIter(BaseNodeIterator[T]):
    """
    Basic iterator which will runs next-calls eagerly
    """

    def __init__(self, base_node: BaseNode[T], initial_state: Optional[Dict[str, Any]]):
        self.base_node = base_node
        self._started = False
        self._finished = False
        if initial_state is not None:
            self._it = self.base_node.iterator(initial_state)
        else:
            self._it = self.base_node.iterator(None)

        self._next_val: Optional[T] = None

    def __next__(self) -> T:
        self._started = True
        if self._next_val is not None:
            val = self._next_val
            self._next_val = None
            return val
        try:
            return next(self._it)
        except StopIteration:
            self._finished = True
            raise

    def has_next(self) -> bool:
        if self._next_val is None:
            try:
                self._next_val = next(self._it)
            except StopIteration:
                pass
        return self._next_val is not None

    def started(self) -> bool:
        return self._started

    def finished(self) -> bool:
        return self._finished
