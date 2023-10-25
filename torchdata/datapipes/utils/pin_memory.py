# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections


def pin_memory_fn(data, device=None):
    r"""
    Utility function to move data to pinned memory. If special treatment is needed to move
    the input data to pinned memory, please attach a ``pin_memory`` method to the expected
    data class.
    """
    if hasattr(data, "pin_memory"):  # Including torch.Tensor
        return data.pin_memory(device)
    elif isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, collections.abc.Mapping):
        pinned_data = {k: pin_memory_fn(sample, device) for k, sample in data.items()}
        try:
            return type(data)(pinned_data)  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return pinned_data
    elif isinstance(data, collections.abc.Sequence):
        pinned_data = [pin_memory_fn(sample, device) for sample in data]  # type: ignore[assignment]
        try:
            return type(data)(pinned_data)  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return pinned_data
    else:
        return data
