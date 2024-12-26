# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing.synchronize as python_mp_synchronize
import queue
import threading
from typing import Callable, Optional, Union

import torch
import torch.multiprocessing as mp

from torch._utils import ExceptionWrapper

from torchdata.nodes.base_node import BaseNode, T

from .constants import QUEUE_TIMEOUT


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
            item, idx = in_q.get(block=True, timeout=QUEUE_TIMEOUT)
        except queue.Empty:
            continue

        if isinstance(item, ExceptionWrapper):
            out_q.put((item, idx), block=False)
        elif isinstance(item, StopIteration):
            out_q.put((item, idx), block=False)
        else:
            try:
                y = udf(item)
            except Exception:
                y = ExceptionWrapper(where="in _apply_udf")

            out_q.put((y, idx), block=False)


class IteratorQ(BaseNode[T]):
    """IteratorQ is a node that takes an iterator and emits its outputs.
    It is a wrapper around an iterator that allows it to be used as a node.
    """

    def __init__(self, q, stop_event):
        super().__init__()
        self.q = q
        self.stop_event = stop_event

    def reset(self, initial_state: Optional[dict] = None):
        super().reset(initial_state)

    def next(self):
        while True:
            if self.stop_event.is_set() and self.q.empty():
                break

            try:
                item, idx = self.q.get(block=True, timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            if isinstance(item, ExceptionWrapper):
                item.reraise()
            elif isinstance(item, StopIteration):
                raise item

            return item

    def get_state(self):
        return {}


def _apply_stream(
    worker_id: int,
    in_q: Union[queue.Queue, mp.Queue],
    out_q: Union[queue.Queue, mp.Queue],
    udf: BaseNode,
    stop_event: Union[threading.Event, python_mp_synchronize.Event],
    first_node: BaseNode,
):
    """_apply_udf assumes in_q emits tuples of (x, idx) where x is the
    payload, idx is the index of the result, potentially used for maintaining
    ordered outputs. For every input it pulls, a tuple (y, idx) is put on the out_q
    where the output of udf(x), an ExceptionWrapper, or StopIteration (if it pulled
    StopIteration from in_q).
    """
    torch.set_num_threads(1)
    first_node.source = IteratorQ(in_q, stop_event)
    while True:
        try:
            item = next(udf)
        except StopIteration as e:
            item = e
        except Exception:
            item = ExceptionWrapper(where="in _apply_stream")

        # if stop_event.is_set() and in_q.empty():
        #     break

        # try:
        #     item, idx = in_q.get(block=True, timeout=QUEUE_TIMEOUT)
        # except queue.Empty:
        #     continue

        out_q.put((item, None), block=False)
        # if isinstance(item, ExceptionWrapper):
        #     out_q.put((item, idx), block=False)
        # elif isinstance(item, StopIteration):
        #     out_q.put((item, idx), block=False)
        # else:
        #     # try:
        #     #     y = udf(item)
        #     # except Exception:
        #     #     y = ExceptionWrapper(where="in _apply_udf")

        #     out_q.put((item, idx), block=False)
