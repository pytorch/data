# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Iterator, TypeVar


T = TypeVar("T", covariant=True)


class BaseNode(Iterable[T]):
    def iterator(self) -> Iterator[T]:
        """Override this method to implement the iterator.
        Iterators are expected to raise StopIteration to signal
        end of iteration, so they can be used in for loops.
        Generators just need to return, as usual.
        """
        raise NotImplementedError()

    def __iter__(self) -> "_EagerIter[T]":
        return _EagerIter(self)


class _EagerIter(Iterator[T]):
    """
    Basic iterator which will runs next-calls eagerly
    """

    def __init__(self, parent: BaseNode[T]):
        self.parent = parent
        self.it = self.parent.iterator()

    def __next__(self):
        return next(self.it)
