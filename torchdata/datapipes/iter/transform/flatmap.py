# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, TypeVar

from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import check_lambda_fn, DILL_AVAILABLE

if DILL_AVAILABLE:
    import dill

    dill.extend(use_dill=False)

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe("flatmap")
class FlatMapperIterDataPipe(IterDataPipe[T_co]):
    r"""
    Applies a function over each item from the source DataPipe, then
    flattens the outputs to a single, unnested IterDataPipe (functional name: ``flatmap``).

    Note:
        The output from ``fn`` must be a Sequence. Otherwise, an error will be raised.

    Args:
        datapipe: Source IterDataPipe
        fn: the function to be applied to each element in the DataPipe, the output must be a Sequence
    """

    def __init__(self, datapipe: IterDataPipe, fn: Callable):
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
