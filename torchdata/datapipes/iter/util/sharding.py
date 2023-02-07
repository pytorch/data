# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, Optional, TypeVar

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("sharding_round_robin_dispatch")
class ShardingRoundRobinDispatcherIterDataPipe(IterDataPipe):
    r"""
    Wrapper that indicates the prior ``DataPipe`` graph is non-replicable and will be
    iterated in a separate dispatching process to coordinate data to worker processes
    in a round-robin manner, when multiprocessing takes place
    (functional name: ``sharding_round_robin_dispatch``).

    Args:
        source_datapipe: Iterable DataPipe that will be sharded
        sharding_group_filter: Optional ``SHARDING_PRIORITIES`` value

    Note:
      - ``sharding_group_filter`` only accepts ``SHARDING_PRIORITIES.MULTIPROCESSING`` for now
    """

    def __init__(self, source_datapipe: IterDataPipe, sharding_group_filter: Optional[SHARDING_PRIORITIES] = None):
        self.source_datapipe = source_datapipe
        if sharding_group_filter != SHARDING_PRIORITIES.MULTIPROCESSING:
            raise NotImplementedError(
                "`sharding_round_robin_dispatch` currently only supports `SHARDING_PRIORITIES.MULTIPROCESSING`."
                "Please open issue on github for your feature request."
            )
        self.sharding_group_filter = sharding_group_filter

    def __iter__(self) -> Iterator[T_co]:
        yield from self.source_datapipe

    def __len__(self) -> int:
        return len(self.source_datapipe)
