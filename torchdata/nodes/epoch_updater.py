# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional

from torchdata.nodes.base_node import BaseNode, T


class EpochUpdater(BaseNode[T]):
    """A node that updates the epoch of the root node."""

    NUM_YIELDED_KEY = "_num_yielded"
    EPOCH_KEY = "_epoch"
    ROOT_KEY = "_root"

    def __init__(
        self,
        root: BaseNode[T],
        initial_epoch: int = 0,
        epoch_updater_fn: Optional[Callable[[int], int]] = None,
    ):
        super().__init__()
        self.root = root
        self._num_yielded = 0
        self._started = False
        self.epoch = initial_epoch
        self.epoch_updater_fn = epoch_updater_fn or self._default_epoch_updater_fn

    def reset(self, initial_state: Optional[dict] = None) -> None:
        super().reset(initial_state)
        if initial_state is not None:
            self._num_yielded = initial_state[self.NUM_YIELDED_KEY]
            self.epoch = initial_state[self.EPOCH_KEY]
            if hasattr(self.root, "set_epoch"):
                self.root.set_epoch(self.epoch)  # pyre-ignore
                self.root.reset(initial_state[self.ROOT_KEY])
        else:
            self._num_yielded = 0
            if self._started:
                # Don't update epoch unless iterator has started
                self.epoch = self.epoch_updater_fn(self.epoch)
            if hasattr(self.root, "set_epoch"):
                self.root.set_epoch(self.epoch)  # pyre-ignore
            self.root.reset(None)
        self._started = False

    def next(self) -> T:
        self._started = True
        self._num_yielded += 1
        item = next(self.root)
        return item

    def get_state(self) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {
            self.NUM_YIELDED_KEY: self._num_yielded,
            self.EPOCH_KEY: self.epoch,
            self.ROOT_KEY: self.root.state_dict(),
        }
        return state_dict

    @classmethod
    def _default_epoch_updater_fn(cls, epoch: int) -> int:
        return epoch + 1
