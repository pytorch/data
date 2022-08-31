# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

import torch

from torch.utils.data.graph import DataPipe
from torchdata.datapipes.iter.util.cacheholder import _WaitPendingCacheItemIterDataPipe


__all__ = [
    "Adapter",
    "CacheTimeout",
    "Shuffle",
]

assert __all__ == sorted(__all__)


class Adapter:
    @abstractmethod
    def __call__(self, datapipe: DataPipe) -> DataPipe:
        pass


class Shuffle(Adapter):
    r"""
    Shuffle DataPipes adapter allows control over all existing Shuffler (``shuffle``) DataPipes in the graph.

    Args:
        enable: Optional[Boolean] = True
            Shuffle(enable = True) - enables all previously disabled Shuffler DataPipes. If none exists, it will add a new `shuffle` at the end of the graph.
            Shuffle(enable = False) - disables all Shuffler DataPipes in the graph.
            Shuffle(enable = None) - Is noop. Introduced for backward compatibility.

    Example:
        >>>  dp = IterableWrapper(range(size)).shuffle()
        >>>  dl = DataLoader2(dp, [Shuffle(False)])
        >>>  self.assertEqual(list(range(size)), list(dl))
    """

    def __init__(self, enable=True):
        self.enable = enable

    def __call__(self, datapipe: DataPipe) -> DataPipe:
        return torch.utils.data.graph_settings.apply_shuffle_settings(datapipe, shuffle=self.enable)


class CacheTimeout(Adapter):
    r"""
    CacheTimeout DataPipes adapter allows control over timeouts of all existing EndOnDiskCacheHolder (``end_caching``)
    DataPipes in the graph. Usefull when cached pipeline takes to long to execute (ex. slow file downloading).

    Args:
        timeout: int - amount of seconds parallel processes will wait for cached files to appear.

    Example:
        >>>  dl = DataLoader2(dp, [CacheTimeout(600)])
    """

    def __init__(self, timeout=None):
        if timeout is None:
            raise ValueError("timeout should be integer")
        self.timeout = timeout

    def __call__(self, datapipe: DataPipe) -> DataPipe:
        graph = torch.utils.data.graph.traverse(datapipe, only_datapipe=True)
        all_pipes = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        cache_locks = {pipe for pipe in all_pipes if isinstance(pipe, _WaitPendingCacheItemIterDataPipe)}

        for cache_lock in cache_locks:
            cache_lock.set_timeout(self.timeout)

        return datapipe
