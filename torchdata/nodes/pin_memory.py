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
from torchdata.nodes import BaseNode, T

from torchdata.nodes.exception_wrapper import ExceptionWrapper, StartupExceptionWrapper
from torchdata.nodes.map import _SingleThreadedMapper
from torchdata.nodes.snapshot_store import SnapshotStore


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

        src_iter = iter(source)
    except Exception:
        e = StartupExceptionWrapper(where=f"in _pin_memory_loop startup for device {device_id}")
        q.put(e)
        return

    while not stop_event.is_set():
        if not semaphore.acquire(blocking=True, timeout=0.1):
            continue
        try:
            item = next(src_iter)
            item = pin_memory(item, device)
            q.put(item, block=False)
        except StopIteration as e:
            item = e
            q.put(item, block=False)
            break
        except Exception:
            item = ExceptionWrapper(where=f"in _pin_memory_loop for device {device_id}")
            q.put(item, block=False)
            break


class PinMemory(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[T],
        pin_memory_device: str = "",
    ):
        self.source = source
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

        self._it: Optional[_SingleThreadedMapper] = None

    def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]:
        self._it = _SingleThreadedMapper(
            source=self.source,
            prefetch_factor=1,
            worker=functools.partial(
                _pin_memory_loop,
                device_id=self._current_device,
                device=self._pin_memory_device,
            ),
        )
        return self._it

    def get_state(self) -> Dict[str, Any]:
        assert self._it is not None, "get_state() should not be called before iterator()!"
        return self._it.get_state()
        return {self.SOURCE_KEY: self.source.state_dict()}
