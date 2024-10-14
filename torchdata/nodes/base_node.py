from typing import Generic, Iterator, TypeVar

import torch.utils.data


T = TypeVar("T")


class BaseNode(torch.utils.data.IterableDataset, Generic[T]):
    def iterator(self) -> Iterator[T]:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[T]:
        # Do not override this method, override iterator() instead.
        yield from self.iterator()
