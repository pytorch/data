# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from torch.utils.data import IterDataPipe, MapDataPipe


# @functional_datapipe("to_iter_datapipe")  # This line must be kept for .pyi signature parser
class MapToIterConverterIterDataPipe(IterDataPipe):
    """
    Convert a ``MapDataPipe`` to an ``IterDataPipe`` (functional name: ``to_iter_datapipe``). It uses ``indices`` to
    iterate through the ``MapDataPipe``, defaults to ``range(len(mapdatapipe))`` if not given.

    For the opposite converter, use :class:`.IterToMapConverter`.

    Args:
        datapipe: source MapDataPipe with data
        indices: optional list of indices that will dictate how the datapipe will be iterated over

    Example:
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> source_dp = SequenceWrapper(range(10))
        >>> iter_dp = source_dp.to_iter_datapipe()
        >>> list(iter_dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> source_dp2 = SequenceWrapper({'a': 1, 'b': 2, 'c': 3})
        >>> iter_dp2 = source_dp2.to_iter_datapipe(indices=['a', 'b', 'c'])
        >>> list(iter_dp2)
        [1, 2, 3]
    """

    # Note that ``indices`` has ``Optional[List]`` instead of ``Optional[Iterable]`` as type because a generator
    # can be passed in as an iterable, which will complicate the serialization process as we will have
    # to materialize ``indices`` and store it.
    def __init__(self, datapipe: MapDataPipe, indices: Optional[List] = None):
        if not isinstance(datapipe, MapDataPipe):
            raise TypeError(f"MapToIterConverter can only apply on MapDataPipe, but found {type(datapipe)}")
        self.datapipe: MapDataPipe = datapipe
        self.indices = indices if indices else range(len(datapipe))

    def __iter__(self):
        for idx in self.indices:
            yield self.datapipe[idx]

    def __len__(self):
        return len(self.indices)


MapDataPipe.register_datapipe_as_function("to_iter_datapipe", MapToIterConverterIterDataPipe)
