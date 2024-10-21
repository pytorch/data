from typing import Generic, Iterator, TypeVar

import torch.utils.data


T = TypeVar("T")


class BaseNode(torch.utils.data.IterableDataset, Generic[T]):
    def iterator(self) -> Iterator[T]:
        """Override this method to implement the iterator.
        Iterators are expected to raise StopIteration to signal
        end of iteration, so they can be used in for loops.
        Generators just need to return, as usual.
        """
        raise NotImplementedError()

    def __iter__(self) -> "_EagerIter[T]":
        return _EagerIter(self)


class _EagerIter(Iterator[T]):
    """
    Basic iterator which will runs next-calls eagerly
    """

    def __init__(self, parent: BaseNode[T]):
        self.parent = parent
        self.it = self.parent.iterator()

    def __next__(self):
        return next(self.it)
