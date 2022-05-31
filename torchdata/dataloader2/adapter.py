# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

from typing import Callable, Optional

import torch
from torchdata.datapipes.iter import IterDataPipe

__all__ = [
    "Adapter",
    "shuffle",
]

assert __all__ == sorted(__all__)

Adapter = Callable[[IterDataPipe], IterDataPipe]


def shuffle(enable=None) -> Adapter:
    return functools.partial(_shuffle, enable)


def _shuffle(enable: Optional[bool], datapipe: IterDataPipe) -> IterDataPipe:
    datapipe = torch.utils.data.graph_settings.apply_shuffle_settings(datapipe, shuffle=enable)
    return datapipe
