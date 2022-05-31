# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

import torch

from torchdata.datapipes.iter import IterDataPipe

__all__ = [
    "Adapter",
    "Shuffle",
]

assert __all__ == sorted(__all__)


class Adapter:
    @abstractmethod
    def __call__(self, datapipe: IterDataPipe) -> IterDataPipe:
        pass


class Shuffle(Adapter):
    r"""
    Shuffle DataPipes adapter allows control over all existing Shuffler (`shuffle`) DataPipes in the graph.

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

    def __call__(self, datapipe: IterDataPipe) -> IterDataPipe:
        return torch.utils.data.graph_settings.apply_shuffle_settings(datapipe, shuffle=self.enable)
