# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch


def generate_random_scalar_tensor(
    rng: Optional[torch.Generator] = None, dtype: torch.dtype = torch.int64
) -> torch.Tensor:
    return torch.empty((), dtype=dtype).random_(generator=rng)


def generate_random_int(rng: Optional[torch.Generator] = None, dtype: torch.dtype = torch.int64) -> int:
    return int(generate_random_scalar_tensor(rng, dtype).item())
