# pyre-unsafe
import queue
import threading

from typing import Iterator, Optional

import torch
from torch._utils import ExceptionWrapper

from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL

from torch.utils.data._utils.pin_memory import pin_memory
from torchdata.nodes import BaseNode, T

from ._populate_queue import _populate_queue


def _pin_memory_loop(
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    device_id: str,
    done_event: threading.Event,
    device: str,
):
    # this is fork of from torch.utils.data._utils.pin_memory import _pin_memory_loop
    # to remove the index tuples

    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.multiprocessing._set_thread_name("pt_data_pin")

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)

    def do_one_step():
        try:
            data = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper) and not isinstance(data, StopIteration):
            try:
                data = pin_memory(data, device)
            except Exception:
                data = ExceptionWrapper(where=f"in pin memory thread for device {device_id}")
        while not done_event.is_set():
            try:
                out_queue.put(data, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()


class PinMemory(BaseNode[T]):
    def __init__(
        self,
        source: BaseNode[T],
        pin_memory_device: str = "",
    ):
        self.source = source

        self.in_q: queue.Queue = queue.Queue()
        self.out_q: queue.Queue = queue.Queue()
        self.sem = threading.BoundedSemaphore(value=1)

        self._started = False
        self._populate_queue_stop_event = threading.Event()
        self._pin_memory_stop_event = threading.Event()

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

        self.read_thread: Optional[threading.Thread] = None
        self.pin_memory_thread: Optional[threading.Thread] = None

    def iterator(self) -> Iterator[T]:
        if not self._started:
            self._populate_queue_stop_event.clear()
            self._pin_memory_stop_event.clear()

            self.read_thread = threading.Thread(
                target=_populate_queue,
                args=(
                    self.source,
                    self.in_q,
                    self._populate_queue_stop_event,
                    self.sem,
                ),
            )
            self.pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(
                    self.in_q,
                    self.out_q,
                    self.current_device,
                    self._pin_memory_stop_event,
                    self._pin_memory_device,
                ),
            )

            self.read_thread.start()
            self.pin_memory_thread.start()
            self._started = True

        exception: Optional[ExceptionWrapper] = None
        while True:
            try:
                item = self.out_q.get(block=True, timeout=0.1)
            except queue.Empty:
                continue

            self.sem.release()
            if isinstance(item, StopIteration):
                break
            elif isinstance(item, ExceptionWrapper):
                exception = item
                break
            yield item

        self._populate_queue_stop_event.set()
        self._pin_memory_stop_event.set()
        if exception is not None:
            exception.reraise()
        self._shutdown()

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        self._populate_queue_stop_event.set()
        self._pin_memory_stop_event.set()
        if self.read_thread is not None:
            self.read_thread.join(timeout=0.1)
        if self.pin_memory_thread is not None:
            self.pin_memory_thread.join(timeout=0.1)
        self._started = False
        self.read_thread = None
        self.pin_memory_thread = None
