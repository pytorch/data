# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


class StopCriteria:
    """
    Stopping criteria for the dataset samplers.

    1) CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED: Stop once the last unseen dataset is exhausted.
       All datasets are seen at least once. In certain cases, some datasets may be
       seen more than once when there are still non-exhausted datasets.

    2) ALL_DATASETS_EXHAUSTED: Stop once all have the datasets are exhausted. Each
       dataset is seen exactly once. No wraparound or restart will be performed.

    3) FIRST_DATASET_EXHAUSTED: Stop when the first dataset is exhausted.

    4) CYCLE_FOREVER: Cycle through the datasets by reinitializing each exhausted source nodes.
       This is useful when trainer want control over certain number of steps instead of epochs.
    """

    CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED = "CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED"
    ALL_DATASETS_EXHAUSTED = "ALL_DATASETS_EXHAUSTED"
    FIRST_DATASET_EXHAUSTED = "FIRST_DATASET_EXHAUSTED"
    CYCLE_FOREVER = "CYCLE_FOREVER"
