import threading
from typing import Generic, Iterator, TypeVar

import torch.utils.data
from torch._utils import ExceptionWrapper


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
        # Do not override this method, override iterator() instead.
        for x in self.iterator():
            if isinstance(x, ExceptionWrapper) and threading.main_thread() is threading.current_thread():
                # We re-raise exceptions as early as possible once we're in the main thread
                x.reraise()
            yield x
