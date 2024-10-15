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

    def __iter__(self) -> Iterator[T]:
        yield from self.iterator()
