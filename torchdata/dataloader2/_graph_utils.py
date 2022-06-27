# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Union

from torch.utils.data.graph import traverse

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe

__all__ = ["DataPipe", "DataPipeGraph", "traverse"]

DataPipe = Union[IterDataPipe, MapDataPipe]
# TODO(VitalyFedyunin): This type is actually confusing, consider renaming to DataPipeGraphTraverse
DataPipeGraph = Dict[DataPipe, "DataPipeGraph"]  # type: ignore[misc]
