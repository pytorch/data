# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.distributed as dist


_HALF_UINT64 = 0x8000000000000000


def dist_share_seed(seed: int, process_group: Optional[dist.ProcessGroup] = None) -> int:
    # Convert uint64 to int64 to prevent overflow for integer Tensor
    seed -= _HALF_UINT64
    shared_seed = torch.tensor(seed, dtype=torch.int64)
    dist.broadcast(shared_seed, src=0, group=process_group)
    # Revert int64 back to uint64
    return int(shared_seed.item()) + _HALF_UINT64
