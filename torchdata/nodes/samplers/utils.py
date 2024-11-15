# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


class StopCriteria:
    """
    Stopping criteria for the dataset samplers.

    1) CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED: Stop once the last unseen dataset is exhausted.
        All datasets are seen at least once. In certain cases, some datasets may be
        seen more than once when there are still non-exhausted datasets.

    2) ALL_DATASETS_EXHAUSTED: Stop once all have the datasets are exhausted. Each
        dataset is seen exactly once. No wraparound or restart will be performed.

    3) FIRST_DATASET_EXHAUSTED: Stop when the first dataset is exhausted.
    """

    CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED = "CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED"
    ALL_DATASETS_EXHAUSTED = "ALL_DATASETS_EXHAUSTED"
    FIRST_DATASET_EXHAUSTED = "FIRST_DATASET_EXHAUSTED"


def _get_worker_seed(seed: int, g_worker: torch.Generator) -> int:
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    num_workers = worker_info.num_workers if worker_info is not None else 1
    # g_worker = torch.Generator()
    g_worker.manual_seed(seed * num_workers + worker_id)
    return int(torch.randint(0, 2 ** 32 - 1, size=(1,), generator=g_worker).item())
