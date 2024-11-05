# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import queue
import threading

from typing import Any, Dict, Iterator, Optional, Union

import torch
import torch.multiprocessing

from torch.utils.data._utils.pin_memory import pin_memory
from torchdata.nodes.base_node import BaseNode, T

from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper
from torchdata.nodes.map import _SingleThreadedMapper
from torchdata.nodes.snapshot_store import MonotonicIndex, SnapshotStore


def _pin_memory_loop(
    source: BaseNode,
    q: queue.Queue,
    snapshot_store: SnapshotStore,
    snapshot_frequency: int,
    semaphore: threading.BoundedSemaphore,
    stop_event: threading.Event,
    device_id: Union[int, str],
    device: Optional[str],
):
    # this is fork of from torch.utils.data._utils.pin_memory import _pin_memory_loop
    # to remove the index tuples

    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.

    idx = MonotonicIndex()

    def _put(item, block: bool = True, snapshot: Optional[Dict[str, Any]] = None):
        _idx = idx.get()
        if snapshot:
            snapshot_store.append(snapshot=snapshot, version=_idx)
        q.put((item, _idx), block=block)

    try:
        torch.set_num_threads(1)

        torch.multiprocessing._set_thread_name("pt_data_pin")

        if device == "cuda":
            torch.cuda.set_device(device_id)
        elif device == "xpu":
            torch.xpu.set_device(device_id)  # type: ignore[attr-defined]
        elif device == torch._C._get_privateuse1_backend_name():
            custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
            custom_device_mod.set_device(device_id)

        assert (
            isinstance(snapshot_frequency, int) and snapshot_frequency >= 0
        ), f"snapshot_frequency must be non-negative integer! Got {snapshot_frequency}"
        src_iter = iter(source)
    except Exception:
        e = StartupExceptionWrapper(where=f"in _pin_memory_loop startup for device {device_id}")
        _put(e)
        return

    yielded = 0
    while not stop_event.is_set():
        if not semaphore.acquire(blocking=True, timeout=0.1):
            continue
        try:
            item = next(src_iter)
            item = pin_memory(item, device)
            yielded += 1
            snapshot = None
            if snapshot_frequency > 0 and yielded % snapshot_frequency == 0:
                snapshot = source.state_dict()
            _put(item, block=False, snapshot=snapshot)
        except StopIteration as e:
            item = e
            _put(item, block=False)
            break
        except Exception:
            item = ExceptionWrapper(where=f"in _pin_memory_loop for device {device_id}")
            _put(item, block=False)
            break


class PinMemory(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[T],
        pin_memory_device: str = "",
        snapshot_frequency: int = 1,
    ):
        self.source = source
        self.snapshot_frequency = snapshot_frequency
        self._pin_memory = torch.cuda.is_available()
        if len(pin_memory_device) == 0:
            self._pin_memory_device = None
        else:
            self._pin_memory_device = pin_memory_device

        if self._pin_memory_device == "xpu":
            self._current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
        elif self._pin_memory_device == torch._C._get_privateuse1_backend_name():
            custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
            self._current_device = custom_device_mod.current_device()
        else:
            self._current_device = torch.cuda.current_device()

        self._it: Optional[_SingleThreadedMapper[T]] = None

    def _get_iterator(self, initial_state: Optional[Dict[str, Any]]) -> _SingleThreadedMapper[T]:
        return _SingleThreadedMapper(
            source=self.source,
            prefetch_factor=1,
            worker=functools.partial(
                _pin_memory_loop,
                device_id=self._current_device,
                device=self._pin_memory_device,
            ),
            snapshot_frequency=self.snapshot_frequency,
            initial_state=initial_state,
        )

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        if self._iter_for_state_dict:
            self._iter_for_state_dict = False
        else:
            self._it = self._get_iterator(initial_state)
        assert self._it is not None
        return self._it

    def get_state(self) -> Dict[str, Any]:
        if self._it is None:
            self._it = self._get_iterator(None)
            self._iter_for_state_dict = True
        return self._it.get_state()
