from typing import Any, Dict, Iterator, Optional

from torchdata.nodes.base_node import BaseNode, T


class Loader(BaseNode[T]):
    def __init__(self, root: BaseNode[T], restart_on_stop_iteration: bool = True):
        self.root = root
        self.restart_on_stop_iteration = restart_on_stop_iteration
        self._it = None

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        self._it = self.Iter(self, initial_state)
        return self._it

    def get_state(self) -> Dict[str, Any]:
        if self._it is None:
            iter(self)
        return self._it.get_state()

    class Iter(Iterator[T]):
        ROOT_KEY = "root"

        def __init__(self, parent, initial_state: Optional[Dict[str, Any]]):
            self.root = parent.root
            if initial_state is not None:
                self.root.load_state_dict(
                    initial_state[self.ROOT_KEY],
                    restart_on_stop_iteration=parent.restart_on_stop_iteration,
                )
            self._it = iter(self.root)

        def __iter__(self):
            return self

        def __next__(self) -> T:
            return next(self._it)

        def get_state(self) -> Dict[str, Any]:
            return {self.ROOT_KEY: self.root.state_dict()}
