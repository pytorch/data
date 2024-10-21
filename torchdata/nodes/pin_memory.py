# pyre-unsafe
import queue
import threading

from typing import Iterator, Optional

import torch
import torch.multiprocessing
from torch._utils import ExceptionWrapper

from torch.utils.data._utils.pin_memory import pin_memory
from torchdata.nodes import BaseNode, T


def _pin_memory_loop(
    source: queue.Queue,
    q: queue.Queue,
    semaphore: threading.BoundedSemaphore,
    stop_event: threading.Event,
    device_id: str,
    device: str,
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
        e = ExceptionWrapper(where=f"in _pin_memory_loop startup for device {device_id}")
        q.put(e)
        return

    while not stop_event.is_set():
        if not semaphore.acquire(blocking=True, timeout=0.1):
            continue
        try:
            x = next(src_iter)
            x = pin_memory(x, device)
        except StopIteration as e:
            q.put(e)
            break
        except Exception:
            x = ExceptionWrapper(where=f"in _pin_memory_loop for device {device_id}")

        while not stop_event.is_set():
            q.put(x, block=False)


class PinMemory(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[T],
        pin_memory_device: str = "",
    ):
        self.source = source

        self._out_q: queue.Queue = queue.Queue()
        self._sem = threading.BoundedSemaphore(value=1)

        self._started = False
        self._stop_event = threading.Event()

        self._pin_memory = torch.cuda.is_available()
        if len(pin_memory_device) == 0:
            self._pin_memory_device = None
        else:
            self._pin_memory_device = pin_memory_device

        if self._pin_memory_device == "xpu":
            self.current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
        elif self._pin_memory_device == torch._C._get_privateuse1_backend_name():
            custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
            self.current_device = custom_device_mod.current_device()
        else:
            self.current_device = torch.cuda.current_device()  # choose cuda for default

        self._thread: Optional[threading.Thread] = None

    def iterator(self) -> Iterator[T]:
        if not self._started:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=_pin_memory_loop,
                args=(
                    self.source,
                    self._out_q,
                    self._sem,
                    self._stop_event,
                    self.current_device,
                    self._pin_memory_device,
                ),
            )

            self._thread.start()
            self._started = True

        exception: Optional[ExceptionWrapper] = None
        while True:
            try:
                item = self._out_q.get(block=True, timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, StopIteration):
                self._sem.release()
                break
            elif isinstance(item, ExceptionWrapper):
                exception = item
                if "_pin_memory_loop startup" not in exception.where:
                    # We don't need to release for startup exceptions
                    self._sem.release()
                break
            yield item

        self._stop_event.set()
        if exception is not None:
            exception.reraise()
        self._shutdown()

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)
        self._started = False
        self._thread = None
