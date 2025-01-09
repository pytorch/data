# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import queue

import threading
from concurrent.futures import Executor
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T", covariant=True)


class BaseOperator(Generic[T]):
    def __init__(self, *args, **kwargs):
        self._q = queue.Queue()
        self._executor: Optional[Executor] = None
        self.lock = threading.Lock()
        self.__initialized = False

    def process(self, item) -> T:
        """Subclasses must implement this method, instead of ``__next__``. Should only be called by BaseNode.

        Returns:
            T - the next value in the sequence, or throw StopIteration
        """
        raise NotImplementedError(type(self))

    def get_source(self) -> "Optional[BaseOperator[T]]":
        raise NotImplementedError(type(self))

    def set_executor(self, executor: Executor):
        self._executor = executor
        src = self.get_source()
        if src is not None:
            src.set_executor(executor)

    def callback_wrapper(self, item: Any, callback):
        assert self._executor is not None
        result = self.process(item)
        if result is None:
            self._executor.submit(
                self.get_source().get,
                callback=functools.partial(self.callback_wrapper, callback=callback),
            )
        else:
            if isinstance(result, list):
                for r in result[1:]:
                    self._q.put(r)
                result = result[0]
            self._executor.submit(callback, result)

    def get(self, callback):
        assert self._executor is not None
        with self.lock:
            if self._q.empty():
                self._executor.submit(
                    self.get_source().get,
                    callback=functools.partial(self.callback_wrapper, callback=callback),
                )
            else:
                self._executor.submit(callback, self._q.get())

    def reset(self, initial_state: Optional[dict] = None):
        """Resets the iterator to the beginning, or to the state passed in by initial_state.

        Reset is a good place to put expensive initialization, as it will be lazily called when ``next()`` or ``state_dict()`` is called.
        Subclasses must call ``super().reset(initial_state)``.

        Args:
            initial_state: Optional[dict] - a state dict to pass to the node. If None, reset to the beginning.
        """

        self.__initialized = True

    def get_state(self) -> Dict[str, Any]:
        """Subclasses must implement this method, instead of ``state_dict()``. Should only be called by BaseNode.

        Returns:
            Dict[str, Any] - a state dict that may be passed to ``reset()`` at some point in the future
        """
        raise NotImplementedError(type(self))

    def state_dict(self) -> Dict[str, Any]:
        """Get a state_dict for this BaseNode.

        Returns:
            Dict[str, Any] - a state dict that may be passed to ``reset()`` at some point in the future.
        """
        try:
            self.__initialized
        except AttributeError:
            raise NotImplementedError(f"self.__initialized not found, did you call super().__init__()? {type(self)=}")

        if not self.__initialized:
            self.reset(None)
            if not self.__initialized:
                raise NotImplementedError(
                    f"Failed to initialize after .reset(), did you call super().reset() in your .reset() method? {type(self)=}"
                )
        return self.get_state()


class BaseNode(Iterator[T], BaseOperator[T]):
    """BaseNodes are the base class for creating composable dataloading DAGs in ``torchdata.nodes``.

    Most end-users will not iterate over a BaseNode instance directly, but instead
    wrap it in a :class:`torchdata.nodes.Loader` which converts the DAG into a more familiar Iterable.

    .. code-block:: python

        node = MyBaseNodeImpl()
        loader = Loader(node)
        # loader supports state_dict() and load_state_dict()

        for epoch in range(5):
            for idx, batch in enumerate(loader):
                ...

        # or if using node directly:
        node = MyBaseNodeImpl()
        for epoch in range(5):
            node.reset()
            for idx, batch in enumerate(loader):
                ...
    """

    def __init__(self, *args, **kwargs):
        """Subclasses must implement this method and call super().__init__(*args, **kwargs)"""
        super().__init__(*args, **kwargs)
        self.__initialized = False

    def __iter__(self):
        return self

    def next(self) -> T:
        """Subclasses must implement this method, instead of ``__next__``. Should only be called by BaseNode.

        Returns:
            T - the next value in the sequence, or throw StopIteration
        """
        raise NotImplementedError(type(self))

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.__initialized = True

    def get(self, callback):
        assert self._executor is not None
        try:
            item = next(self)
        except StopIteration as e:
            item = e
        return self._executor.submit(callback, item)

    def __next__(self):
        try:
            self.__initialized
        except AttributeError:
            raise NotImplementedError(f"self.__initialized not found, did you call super().__init__()? {type(self)=}")
        if not self.__initialized:
            self.reset(None)
            if not self.__initialized:
                raise NotImplementedError(
                    f"Failed to initialize after .reset(), did you call super().reset() in your .reset() method? {type(self)=}"
                )

        return self.next()
