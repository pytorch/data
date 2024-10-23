# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
import threading
from typing import Callable, Union

import torch
import torch.multiprocessing as mp
from torch import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as python_mp

from torch._utils import ExceptionWrapper


def _apply_udf(
    worker_id: int,
    in_q: Union[queue.Queue, mp.Queue],
    out_q: Union[queue.Queue, mp.Queue],
    udf: Callable,
    stop_event: Union[threading.Event, python_mp.synchronize.Event],
):
    """_apply_udf assumes in_q is emitting tuples of (x, idx) where x is the
    payload, idx is the index of the result, potentially used for maintaining
    ordered outputs
    """
    torch.set_num_threads(1)
    while True:
        if stop_event.is_set() and in_q.empty():
            break

        try:
            x, idx = in_q.get(block=True, timeout=1.0)
        except queue.Empty:
            continue

        if isinstance(x, ExceptionWrapper):
            out_q.put((x, idx))
        elif isinstance(x, StopIteration):
            out_q.put((x, idx))
        else:
            try:
                y = udf(x)
            except Exception:
                y = ExceptionWrapper(where="in _apply_udf")

            out_q.put((y, idx), block=False)
