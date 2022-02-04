# Copyright (c) Facebook, Inc. and its affiliates.
import types
from typing import Optional, Sequence, TypeVar

from torch.utils.data.datapipes.iter.combining import _ChildDataPipe, _ForkerIterDataPipe
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


T = TypeVar("T")


@functional_datapipe("unzip")
class UnZipperIterDataPipe(IterDataPipe[T]):
    r"""
    Takes in a DataPipe of Sequences, unpacks each Sequence, and return the elements in separate DataPipes
    based on their position in the Sequence.

    Note:
        Each sequence within the DataPipe should have the same length, optionally specified by
        the input argument `sequence_length`.

    Args:
        source_datapipe: Iterable DataPipe with sequences of data
        sequence_length: Length of the sequence within the source_datapipe. All elements should have the same length.
            If `None`, it is inferred from the first sequence in the `source_datapipe`.
        buffer_size: this restricts how far ahead the leading child DataPipe can read relative
            to the slowest child DataPipe. Use -1 for the unlimited buffer.
    """

    def __new__(
        cls, source_datapipe: IterDataPipe[Sequence[T]], sequence_length: Optional[int] = None, buffer_size: int = 1000
    ):

        if sequence_length is None:
            it = iter(source_datapipe)
            first_seq = next(it)
            num_instances = len(first_seq)
            # TODO: In the future, `source_datapipe` might need to be properly reset (if multiple iterators
            #       per DataPipe is not allowed)
        else:
            num_instances = sequence_length

        # The implementation basically uses Forker but only yields a specific element within the sequence
        container = _ForkerIterDataPipe(source_datapipe, num_instances, buffer_size)

        def _get_generator_by_instance(self, instance_id):
            for x in self.main_datapipe.get_next_element_by_instance(instance_id):
                yield x[instance_id]

        child_dps = [_ChildDataPipe(container, i) for i in range(num_instances)]
        for dp in child_dps:
            setattr(dp, "get_generator_by_instance", types.MethodType(_get_generator_by_instance, dp))  # noqa: B010

        return child_dps
