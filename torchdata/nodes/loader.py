from typing import Any, Dict, Generic, Optional

from torchdata.nodes.base_node import BaseNode, T


class Loader(Generic[T]):
    def __init__(self, root: BaseNode[T], restart_on_stop_iteration: bool = True):
        super().__init__()
        self.root = root
        self.restart_on_stop_iteration = restart_on_stop_iteration
        self._next_iter_state_dict: Optional[Dict[str, Any]] = None
        self._it: Optional[LoaderIterator[T]] = None

    def __iter__(self):
        if self._it is None:
            self._it = LoaderIterator(self)

        if self._next_iter_state_dict is not None:
            self._it.reset(initial_state=self._next_iter_state_dict)
            self._next_iter_state_dict = None
            if self.restart_on_stop_iteration and not self._it.has_next():
                self._it.reset(None)
        else:
            self._it.reset(None)

        return self._it

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._next_iter_state_dict = state_dict

    def state_dict(self) -> Dict[str, Any]:
        if self._it is None:
            iter(self)
        return self._it.state_dict()  # type:ignore[union-attr]


class LoaderIterator(BaseNode[T]):
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
        self._saved_currect_state_dict: Optional[Dict[str, Any]] = None
        self._num_yielded = 0

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self.root.reset(initial_state[self.ROOT_KEY])
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
        else:
            self.root.reset(None)
            self._num_yielded = 0
        self._cached_item = None

    def has_next(self) -> bool:
        if self._cached_item is None:
            try:
                self._saved_currect_state_dict = self.state_dict()
                self._cached_item = next(self)
            except StopIteration:
                pass
        return self._cached_item is not None

    def next(self):
        if self._cached_item is not None:
            item = self._cached_item
            self._cached_item = None
            self._saved_currect_state_dict = None
        else:
            item = next(self.root)
        self._num_yielded += 1
        return item

    def get_state(self) -> Dict[str, Any]:
        if self._saved_currect_state_dict is not None:
            return self._saved_currect_state_dict
        return {
            self.ROOT_KEY: self.root.state_dict(),
            self.NUM_YIELDED_KEY: self._num_yielded,
        }
