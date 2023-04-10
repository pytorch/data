# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

import torch

from torchdata.dataloader2.graph import DataPipe, traverse_dps
from torchdata.datapipes.iter.util.cacheholder import _WaitPendingCacheItemIterDataPipe


__all__ = [
    "Adapter",
    "CacheTimeout",
    "Shuffle",
]

assert __all__ == sorted(__all__)


class Adapter:
    r"""
    Adapter Base Class that follows python Callable protocol.
    """

    @abstractmethod
    def __call__(self, datapipe: DataPipe) -> DataPipe:
        r"""
        Callable function that either runs in-place modification of
        the ``DataPipe`` graph, or returns a new ``DataPipe`` graph.

        Args:
            datapipe: ``DataPipe`` that needs to be adapted.

        Returns:
            Adapted ``DataPipe`` or new ``DataPipe``.
        """
        pass


class Shuffle(Adapter):
    r"""
    Shuffle DataPipes adapter allows control over all existing Shuffler (``shuffle``) DataPipes in the graph.

    Args:
        enable: Optional boolean argument to enable/disable shuffling in the ``DataPipe`` graph. True by default.

            - True: Enables all previously disabled ``ShufflerDataPipes``. If none exists, it will add a new ``shuffle`` at the end of the graph.
            - False: Disables all ``ShufflerDataPipes`` in the graph.
            - None: No-op. Introduced for backward compatibility.

    Example:

    .. testsetup::

        from torchdata.datapipes.iter import IterableWrapper
        from torchdata.dataloader2 import DataLoader2
        from torchdata.dataloader2.adapter import Shuffle

        size = 12

    .. testcode::

        dp = IterableWrapper(range(size)).shuffle()
        dl = DataLoader2(dp, [Shuffle(False)])
        assert list(range(size)) == list(dl)
    """

    def __init__(self, enable=True):
        self.enable = enable

    def __call__(self, datapipe: DataPipe) -> DataPipe:
        return torch.utils.data.graph_settings.apply_shuffle_settings(datapipe, shuffle=self.enable)


class CacheTimeout(Adapter):
    r"""
    CacheTimeout DataPipes adapter allows control over timeouts of all existing EndOnDiskCacheHolder (``end_caching``)
    in the graph. Useful when cached pipeline takes too long to execute (ex. slow file downloading).

    Args:
        timeout: int - amount of seconds parallel processes will wait for cached files to appear.

    Example:

    .. testsetup::

        from torchdata.datapipes.iter import IterableWrapper
        from torchdata.dataloader2 import DataLoader2
        from torchdata.dataloader2.adapter import CacheTimeout

        size = 12

    .. testcode::

        dp = IterableWrapper(range(size)).shuffle()
        dl = DataLoader2(dp, [CacheTimeout(600)])
    """

    def __init__(self, timeout=None):
        if timeout is None:
            raise ValueError("timeout should be integer")
        self.timeout = timeout

    def __call__(self, datapipe: DataPipe) -> DataPipe:
        graph = traverse_dps(datapipe)
        all_pipes = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        cache_locks = {pipe for pipe in all_pipes if isinstance(pipe, _WaitPendingCacheItemIterDataPipe)}

        for cache_lock in cache_locks:
            cache_lock.set_timeout(self.timeout)

        return datapipe
