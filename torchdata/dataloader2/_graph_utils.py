# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Union

from torch.utils.data.graph import traverse
from torchdata.datapipes.iter import IterDataPipe, MapDataPipe

__all__ = ["DataPipeGraph", "traverse"]

DataPipe = Union[IterDataPipe, MapDataPipe]
DataPipeGraph = Dict[IterDataPipe, "DataPipeGraph"]  # type: ignore[misc]
