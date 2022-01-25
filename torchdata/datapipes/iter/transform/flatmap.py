# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data import DataChunk
from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import DILL_AVAILABLE, check_lambda_fn

from typing import TypeVar

if DILL_AVAILABLE:
    import dill
    dill.extend(use_dill=False)

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("flatmap")
class FlatMapperIterDataPipe(IterDataPipe[DataChunk[T_co]]):
    r""":class:`FlatMapperIterDataPipe`.

    Iterable DataPipe which applies a structure-changing function to an IterableDataPipe
    flattens to a single unnested IterableDataPipe.

    Args:
        datapipe: Iterable datapipe containing iterable datapipes to which the function is applied
        fn: the function to be applied to each of the ``inner" datapipes
    """
    def __init__(self, datapipe, fn):
        self.datapipe = datapipe

        check_lambda_fn(fn)
        self.fn = fn  # type: ignore[assignment]

    def __iter__(self):
        for e in self.datapipe:
            yield from self.fn(e)

    def __len__(self):
        raise TypeError(f"{type(self).__name__}'s length relies on the output of its function.")

    def __getstate__(self):
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(self)

        if DILL_AVAILABLE:
            dill_function = dill.dumps(self.fn)
        else:
            dill_function = self.fn
        state = (
            self.datapipe,
            dill_function,
        )
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            dill_function,
        ) = state
        if DILL_AVAILABLE:
            self.fn = dill.loads(dill_function)  # type: ignore[assignment]
        else:
            self.fn = dill_function  # type: ignore[assignment]
