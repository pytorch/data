import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generic, Optional

from torchdata.nodes.base_node import BaseNode, T


class Loader(Generic[T]):
    """Wraps the root BaseNode (an iterator) and provides a stateful iterable interface.

    The state of the last-returned iterator is returned by the state_dict() method, and can be
    loaded using the load_state_dict() method.

    Args:
        root (BaseNode[T]): The root node of the data pipeline.
        restart_on_stop_iteration (bool): Whether to restart the iterator when it reaches the end. Default is True
    """

    def __init__(
        self,
        root: BaseNode[T],
        restart_on_stop_iteration: bool = True,
        num_workers: int = 1,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.root = root
        self.restart_on_stop_iteration = restart_on_stop_iteration
        self._next_iter_state_dict: Optional[Dict[str, Any]] = None
        self._it: Optional[LoaderIterator[T]] = None
        # Tracks whether an iterator was created solely for getting a state_dict, in which case
        # we don't want to reset the iterator. Consider these two cases, which should behave the same
        # it = iter(loader)
        # sd = loader.state_dict()  # No extra __iter__ call as _it already exists
        # for _ in it: ...
        # --------
        # sd = loader.state_dict()  # Calls __iter__ since _it is None
        # it = iter(loader)  # We don't want to reset the iterator here again
        # for _ in it: ...
        self._iter_for_state_dict: bool = False
        self.num_workers = 1
        self.prefetch_factor = prefetch_factor
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self.root.set_executor(self._executor)

    def __iter__(self):
        if self._it is None:
            self._it = LoaderIterator(self)
        elif self._iter_for_state_dict:
            self._iter_for_state_dict = False
            return self._it  # This was already pre-called to get a state dict

        if self._next_iter_state_dict is not None:
            self._it.reset(initial_state=self._next_iter_state_dict)
            self._next_iter_state_dict = None
            if self.restart_on_stop_iteration and not self._it.has_next():
                self._it.reset(None)
        else:
            self._it.reset(None)

        return self._it

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads a state_dict which will be used to initialize the next iter() requested
        from this loader.

        Args:
            state_dict (Dict[str, Any]): The state_dict to load. Should be generated from a call to state_dict().
        """
        self._next_iter_state_dict = state_dict

    def state_dict(self) -> Dict[str, Any]:
        """Returns a state_dict which can be passed to load_state_dict() in the future to
        resume iteration.

        The state_dict will come from the iterator returned by the most recent call to iter().
        If no iterator has been created, a new iterator will be created and the state_dict returned from it.
        """
        if self._it is None:
            iter(self)
            self._iter_for_state_dict = True
        return self._it.state_dict()  # type:ignore[union-attr]


class LoaderIterator(BaseNode[T]):
    """An iterator class that wraps a root node and works with the Loader class.

    The LoaderIterator object saves state of the underlying root node, and calls reset on the root node when
    the iterator is exhausted or on a reset call. We look one step ahead to determine if the iterator is exhausted.
    The state of the iterator is saved in the state_dict() method, and can be loaded on reset calls.

    Args:
        loader (Loader[T]): The loader object that contains the root node.
    """

    NUM_YIELDED_KEY = "num_yielded"
    ROOT_KEY = "root"

    def __init__(
        self,
        loader: Loader[T],
    ):
        super().__init__()
        self.loader = loader
        self.root = loader.root
        self._cached_item = None
        self._cached_state_dict: Optional[Dict[str, Any]] = None
        self._num_yielded = 0
        self._q = queue.Queue()

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self.root.reset(initial_state[self.ROOT_KEY])
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
        else:
            self.root.reset(None)
            self._num_yielded = 0
        self._cached_item = None

        for _ in range(self.loader.prefetch_factor):
            self._prefetch()

    def has_next(self) -> bool:
        if self._cached_item is None:
            try:
                # Cache the current state dict
                self._cached_state_dict = self.state_dict()
                # Load and save the next item
                self._cached_item = self._get_next()  # next(self)
            except StopIteration:
                pass
        return self._cached_item is not None

    def callback(self, val: Any):
        self._q.put(val)

    def _prefetch(self):
        self.loader._executor.submit(self.root.get, callback=self.callback)

    def _get_next(self) -> T:
        while True:
            try:
                item = self._q.get(timeout=1)
            except queue.Empty:
                continue
            if isinstance(item, StopIteration):
                raise StopIteration
            self._prefetch()
            return item

    def next(self):
        if self._cached_item is not None:
            item = self._cached_item
            self._cached_item = None
            self._cached_state_dict = None
        else:
            item = self._get_next()  # next(self.root)
        self._num_yielded += 1
        return item

    def get_state(self) -> Dict[str, Any]:
        if self._cached_state_dict is not None:
            return self._cached_state_dict
        return {
            self.ROOT_KEY: self.root.state_dict(),
            self.NUM_YIELDED_KEY: self._num_yielded,
        }
