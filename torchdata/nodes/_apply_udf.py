import queue

import torch

from torch._utils import ExceptionWrapper


def _apply_udf(worker_id, in_q, out_q, in_order, udf, stop_event):
    torch.set_num_threads(1)
    while True:
        if stop_event.is_set() and in_q.empty():
            break
        else:
            print(worker_id, stop_event, in_q)

        try:  # TODO: implement in-order execution
            x = in_q.get(block=True, timeout=1.0)
        except queue.Empty:
            continue
        if isinstance(x, ExceptionWrapper):
            out_q.put(x)
        elif isinstance(x, StopIteration):
            out_q.put(x)

        try:
            y = udf(x)
        except Exception:
            y = ExceptionWrapper(where="in _apply_udf")

        out_q.put(y, block=False)
