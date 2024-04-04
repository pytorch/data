from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Optional[Dict[str, Any]]:
        ...

    def load_state_dict(self, state_dict: Optional[Dict[str, Any]]) -> None:
        ...
