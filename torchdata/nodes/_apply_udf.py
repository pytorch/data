import queue

import torch

from torch._utils import ExceptionWrapper


def _apply_udf(worker_id, in_q, out_q, sem, in_order, udf, stop_event):
    torch.set_num_threads(1)
    while True:
        if stop_event.is_set() and in_q.empty():
            break

        try:  # TODO: implement in-order execution
            x = in_q.get(block=True, timeout=5.0)
        except queue.Empty:
            continue
        if isinstance(x, ExceptionWrapper):
            print("Got exception wrapper from in_q", x)
            out_q.put(x)
            stop_event.set()
            break
        try:
            y = udf(x)
        except Exception:
            y = ExceptionWrapper(where="in _apply_udf")
            print("Hit exception", y)

        out_q.put(y, block=False)
