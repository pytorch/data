from typing import Iterator

import torch.utils.data


class BaseNode(torch.utils.data.IterableDataset):
    def iterator(self) -> Iterator[T]:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[T]:
        # Do not override this method, override iterator() instead.
        yield from self.iterator()
