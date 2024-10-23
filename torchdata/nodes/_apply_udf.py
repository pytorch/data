# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing.synchronize as python_mp_synchronize
import queue
import threading
from typing import Callable, Union

import torch
import torch.multiprocessing as mp

from torch._utils import ExceptionWrapper


def _apply_udf(
    worker_id: int,
    in_q: Union[queue.Queue, mp.Queue],
    out_q: Union[queue.Queue, mp.Queue],
    udf: Callable,
    stop_event: Union[threading.Event, python_mp_synchronize.Event],
):
    """_apply_udf assumes in_q emits tuples of (x, idx) where x is the
    payload, idx is the index of the result, potentially used for maintaining
    ordered outputs. For every input it pulls, a tuple (y, idx) is put on the out_q
    where the output of udf(x), an ExceptionWrapper, or StopIteration (if it pulled
    StopIteration from in_q).
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
