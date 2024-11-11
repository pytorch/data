from typing import Any, Dict, Iterator, Optional

from torchdata.nodes.base_node import BaseNode, T


class DataLoader(BaseNode[T]):
    ROOT_KEY = "root"

    def __init__(self, root: BaseNode[T], restart_on_stop_iteration: bool = True):
        self.root = root
        self.restart_on_stop_iteration = restart_on_stop_iteration

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        if initial_state is not None:
            self.root.load_state_dict(
                initial_state[self.ROOT_KEY],
                restart_on_stop_iteration=self.restart_on_stop_iteration,
            )
        yield from self.root

    def get_state(self) -> Dict[str, Any]:
        return {self.ROOT_KEY: self.root.get_state()}
