# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T", covariant=True)


class BaseNode(Iterator[T]):
    """BaseNodes are iterators. They have the following **public** interface:

    * reset(initial_state: Optional[dict] = None) - resets iterator to either initial_state or beginning if None is passed
    * state_dict() -> Dict[str, Any] - returns a state_dict that may be passed to reset() at some point in the future
    * __next__() -> T - users should call next(my_instance) on the iterator in order to iterate through it.

    Base nodes also work in for loops as usual, if they are wrapped with an iter.
    They can also be used directly, eg when composing BaseNodes, with a slight modification eg:
    ```python
    node = MyBaseNodeImpl()
    loader = Loader(node)
    # loader also supports state_dict() and load_state_dict()
    for epoch in range(5):
        for idx, batch in enumerate(loader):
            ...

    # or if using node directly:
    node = MyBaseNodeImpl()
    for epoch in range(5):
        node.reset()
        for idx, batch in enumerate(loader):
            ...
    ```

    Subclasses of base node must implement the following methods:

    * __init__() - must call super().__init__()
    * reset(initial_state: Optional[dict]=None) - As above. Reset is a good place to put expensive
        initialization, as it will be lazily called when next() or state_dict() is called.
        Must call super().reset(initial_state)
    * next() -> T - logic for returning the next value in the sequence, or throw StopIteration
    * get_state(self) -> dict: returns a dictionary representing state that may be passed to reset()

    """

    def __init__(self, *args, **kwargs):
        self.__initialized = False

    def __iter__(self):
        return self

    def reset(self, initial_state: Optional[dict] = None):
        self.__initialized = True

    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError(type(self))

    def next(self) -> T:
        raise NotImplementedError(type(self))

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

    def state_dict(self) -> Dict[str, Any]:
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
