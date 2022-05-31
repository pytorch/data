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
    def __init__(self, enable):
        self.enable = enable

    def __call__(self, datapipe: IterDataPipe) -> IterDataPipe:
        return torch.utils.data.graph_settings.apply_shuffle_settings(datapipe, shuffle=self.enable)
