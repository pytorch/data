# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, TypeVar

from torch.utils.data.datapipes.iter.combining import _ChildDataPipe, _ForkerIterDataPipe
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


T = TypeVar("T")


@functional_datapipe("unzip")
class UnZipperIterDataPipe(IterDataPipe[T]):
    r"""
    Takes in a DataPipe of Sequences, unpacks each Sequence, and return the elements in separate DataPipes
    based on their position in the Sequence (functional name: ``unzip``). The number of instances produced equals to
    the sequence length minus the number of columns to skip.

    Note:
        Each sequence within the DataPipe should have the same length, specified by
        the input argument `sequence_length`.

    Args:
        source_datapipe: Iterable DataPipe with sequences of data
        sequence_length: Length of the sequence within the source_datapipe. All elements should have the same length.
        buffer_size: this restricts how far ahead the leading child DataPipe can read relative
            to the slowest child DataPipe. Use -1 for the unlimited buffer.
        columns_to_skip: optional indices of columns that the DataPipe should skip (each index should be
            an integer from 0 to sequence_length - 1)

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([(i, i + 10, i + 20) for i in range(3)])
        >>> dp1, dp2, dp3 = source_dp.unzip(sequence_length=3)
        >>> list(dp1)
        [0, 1, 2]
        >>> list(dp2)
        [10, 11, 12]
        >>> list(dp3)
        [20, 21, 22]
    """

    def __new__(
        cls,
        source_datapipe: IterDataPipe[Sequence[T]],
        sequence_length: int,
        buffer_size: int = 1000,
        columns_to_skip: Optional[Sequence[int]] = None,
    ):
        if columns_to_skip is None:
            instance_ids = list(range(sequence_length))
        else:
            skips = set(columns_to_skip)
            instance_ids = [i for i in range(sequence_length) if i not in skips]

        if len(instance_ids) == 0:
            raise RuntimeError(
                "All instances are being filtered out in UnZipperIterDataPipe. Please check"
                "the input `sequence_length` and `columns_to_skip`."
            )

        # The implementation basically uses Forker but only yields a specific element within the sequence
        container = _UnZipperIterDataPipe(source_datapipe, sequence_length, buffer_size)  # type: ignore[arg-type]
        return [_ChildDataPipe(container, i) for i in instance_ids]


class _UnZipperIterDataPipe(_ForkerIterDataPipe):
    def get_next_element_by_instance(self, instance_id: int):
        for return_val in super().get_next_element_by_instance(instance_id):
            yield return_val[instance_id]
