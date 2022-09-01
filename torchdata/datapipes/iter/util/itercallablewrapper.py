# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata.datapipes.iter import IterDataPipe


class IterCallableWrapperIterDataPipe(IterDataPipe):
    r"""
    Given a callable that returns an iterable object, this IterDataPipe
    creates the object and yields from it for each iteration.

    This is different from :class:`.IterableWrapper` in that the callable is invoked once
    for every new iteration. Whereas the ``iterable`` in :class:`.IterableWrapper` is
    stored (possibly deep copied) once and re-read for each iteration; :class:`.IterableWrapper`
    may be empty in the second iteration if ``iterable`` is an iterator.

    Args:
        callable: serializable Callable that returns an Iterable object

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, IterCallableWrapper
        >>> from itertools import compress, dropwhile
        >>> from functools import partial
        >>> data_dp = IterableWrapper(range(10))
        >>> selector_dp = IterableWrapper([1, 0] * 5)
        >>> compress_fn = partial(compress, data_dp, selector_dp)
        >>> compress_dp = IterCallableWrapper(compress_fn)
        >>> list(compress_dp)
        [0, 2, 4, 6, 8]
        >>> list(compress_dp)  # Repeat without issue in the second iteration
        [0, 2, 4, 6, 8]
        >>> dropwhile_dp = IterCallableWrapper(partial(dropwhile, lambda x: x < 5, data_dp))
        >>> list(dropwhile_dp)
        [5, 6, 7, 8, 9]
    """

    def __init__(self, callable):
        self.callable = callable

    def __iter__(self):
        yield from self.callable()
