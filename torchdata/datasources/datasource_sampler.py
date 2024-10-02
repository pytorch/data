# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Union

class SamplerType:
    IN_ORDER = "IN_ORDER"
    PSEUDORANDOM = "PSEUDORANDOM"

@dataclass
class SamplerSettings:
    sampler_type: Union[SamplerType, str] = SamplerType.IN_ORDER
    seed: int = 13
    limit_per_rank: Optional[int] = None
    num_bundles_for_shuffle: int = 4
    drop_last: bool = True
    wrap_around: bool = True
    force_contiguous: bool = False

    def __post_init__(self):
        if self.sampler_type not in (
            SamplerType.IN_ORDER,
            SamplerType.PSEUDORANDOM,
        ):
            raise ValueError(
                f"Unknown SamplerType {self.sampler_type}!",
            )

        if self.limit_per_rank is None or self.limit_per_rank < 0:
            self.limit_per_rank = float("inf")
