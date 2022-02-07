# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Iterable, Optional, Sequence, TypeVar

from torch.utils.data.datapipes.iter.combining import _ChildDataPipe, _ForkerIterDataPipe
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


T = TypeVar("T")


@functional_datapipe("unzip")
class UnZipperIterDataPipe(IterDataPipe[T]):
    r"""
    Takes in a DataPipe of Sequences, unpacks each Sequence, and return the elements in separate DataPipes
    based on their position in the Sequence. The number of instances produced equals to the sequence legnth
    minus the number of columns to skip.

    Note:
        Each sequence within the DataPipe should have the same length, specified by
        the input argument `sequence_length`.

    Args:
        source_datapipe: Iterable DataPipe with sequences of data
        sequence_length: Length of the sequence within the source_datapipe. All elements should have the same length.
        buffer_size: this restricts how far ahead the leading child DataPipe can read relative
            to the slowest child DataPipe. Use -1 for the unlimited buffer.
        columns_to_skip:

    """

    def __new__(
        cls,
        source_datapipe: IterDataPipe[Sequence[T]],
        sequence_length: int,
        buffer_size: int = 1000,
        columns_to_skip: Optional[Iterable[int]] = None,
    ):
        if columns_to_skip is None:
            instance_ids = list(range(sequence_length))
        else:
            skips = set(columns_to_skip)
            instance_ids = [i for i in range(sequence_length) if i not in skips]

        # The implementation basically uses Forker but only yields a specific element within the sequence
        container = _ForkerIterDataPipe(source_datapipe, sequence_length, buffer_size)
        return [_UnZipperChildDataPipe(container, i) for i in instance_ids]


class _UnZipperChildDataPipe(_ChildDataPipe):
    def get_generator_by_instance(self, instance_id: int):
        for x in self.main_datapipe.get_next_element_by_instance(instance_id):
            yield x[instance_id]
