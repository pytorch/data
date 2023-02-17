# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import threading
import time

from itertools import zip_longest

import torch

from torch.utils.data import IterDataPipe, MapDataPipe
from torchdata.dataloader2 import communication
from torchdata.dataloader2.graph._serialization import extract_wrapper

try:
    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

__all__ = [
    "DataPipeToQueuesLoop",
    "CreateProcessForDataPipeline",
    "CreateProcessForMultipleDataPipelines",
    "CreateThreadForDataPipeline",
]


class _ResetCounter:
    exp_cnt: int
    cnt: int
    _reached: bool

    def __init__(self, exp_cnt: int):
        self.exp_cnt = exp_cnt
        self.cnt = 0
        self._reached = False

    def increment(self) -> None:
        self.cnt += 1
        assert self.cnt <= self.exp_cnt

    def is_reached(self) -> bool:
        if self.cnt == self.exp_cnt:
            self._reached = True
        return self._reached

    def reset(self) -> None:
        if self._reached:
            self._reached = False
            self.cnt = 0


def MultipleDataPipesToQueuesLoop(source_datapipes, req_queues, res_queues, name, call_on_process_init=None):
    r"""
    Set the appropriate pipes and protocol server type, and create a loop over multiple datapipes
    with the protocol server in a non-blocking manner.
    """
    assert call_on_process_init is None, "``MultipleDataPipesToQueuesLoop`` does not support call_on_process_init"
    num_loops = len(source_datapipes)
    assert num_loops == len(req_queues) and num_loops == len(
        res_queues
    ), "``MultipleDataPipesToQueuesLoop`` requires the same number of datapipes, request queues and response queues"

    torch.set_num_threads(1)

    loops = []
    reset_iterator_counter = _ResetCounter(num_loops)

    for source_datapipe, req_queue, res_queue in zip(source_datapipes, req_queues, res_queues):
        # Extract Serialization Wrapper
        source_datapipe = extract_wrapper(source_datapipe)
        loops.append(
            _create_datapipe_queue_loop(
                source_datapipe,
                req_queue,
                res_queue,
                name,
                blocking_request_get=False,
                reset_iterator_counter=reset_iterator_counter,
            )
        )  # Non-blocking request with reset counters

    # Using `zip_longest` to guarantee the process is terminated only when
    # all loops have received `TerminateRequest`
    for _ in zip_longest(*loops):
        # time.sleep to make Python switch context to get/send message in mp.Queue
        # TODO(ejguan): Microbenchmarked a synthetic non-replicable case that sleep perform similar to pass.
        #               A more comprehensive benchmarking in real-world scneario is needed.
        time.sleep(0)


def DataPipeToQueuesLoop(source_datapipe, req_queue, res_queue, name, call_on_process_init=None):
    r"""
    Initialize with the given init function, set the appropriate pipe and protocol server type, and
    create a loop with the protocol server.
    """
    # Extract Serialization Wrapper
    source_datapipe = extract_wrapper(source_datapipe)

    if call_on_process_init is not None:
        call_on_process_init(source_datapipe)

    torch.set_num_threads(1)

    loop = _create_datapipe_queue_loop(source_datapipe, req_queue, res_queue, name, blocking_request_get=True)

    for _ in loop:
        pass


def _create_datapipe_queue_loop(
    source_datapipe,
    req_queue,
    res_queue,
    name,
    blocking_request_get=True,
    reset_iterator_counter=None,
):
    if isinstance(source_datapipe, IterDataPipe):
        pipe_type = communication.iter
        protocol_type = communication.protocol.IterDataPipeQueueProtocolServer
    elif isinstance(source_datapipe, MapDataPipe):
        pipe_type = communication.map  # type: ignore[misc]
        protocol_type = communication.protocol.MapDataPipeQueueProtocolServer  # type: ignore[assignment]
    else:
        raise Exception("Only supports IterDataPipe or MapDataPipe, got", source_datapipe)

    return pipe_type.DataPipeBehindQueues(
        source_datapipe,
        protocol_type(req_queue, res_queue),
        name=name,
        blocking_request_get=blocking_request_get,
        reset_iterator_counter=reset_iterator_counter,
    )


def CreateProcessForDataPipeline(multiprocessing_ctx, datapipe, name, call_on_process_init=None):
    r"""
    Given a DataPipe, creates a new process with ``DataPipeToQueuesLoop`` as target,
    and returns ``(process, req_queue, res_queue)``.
    """
    req_queue = multiprocessing_ctx.Queue()
    res_queue = multiprocessing_ctx.Queue()
    process = multiprocessing_ctx.Process(
        target=DataPipeToQueuesLoop, args=(datapipe, req_queue, res_queue, name, call_on_process_init)
    )
    return process, req_queue, res_queue


def CreateThreadForDataPipeline(datapipe, name):
    r"""
    Given a DataPipe, creates a copy of the DataPipe, starts a new Thread with ``DataPipeToQueuesLoop`` as target,
    and returns ``(process, req_queue, res_queue, new_copied_datapipe)``.
    """
    req_queue = communication.queue.ThreadingQueue()
    res_queue = communication.queue.ThreadingQueue()

    try:
        new_datapipe = pickle.loads(pickle.dumps(datapipe))
    except Exception as pe:
        if HAS_DILL:
            try:
                new_datapipe = dill.loads(dill.dumps(datapipe))
            except Exception as de:
                raise Exception("Unable to dill DataPipe to make thread local copy", de)

        else:
            raise Exception("Unable to pickle DataPipe to make thread local copy (consider installing `dill`)", pe)

    process = threading.Thread(
        target=DataPipeToQueuesLoop, args=(new_datapipe, req_queue, res_queue, name), daemon=True
    )
    return process, req_queue, res_queue, new_datapipe


def CreateProcessForMultipleDataPipelines(multiprocessing_ctx, datapipes, name):
    r"""
    Given a DataPipe, creates a new process with ``MultipleDataPipesToQueuesLoop`` as target,
    and returns ``(process, [req_queue_0, ...], [res_queue_0, ...])``.
    """
    req_queues = []
    res_queues = []
    for _ in datapipes:
        req_queues.append(multiprocessing_ctx.Queue())
        res_queues.append(multiprocessing_ctx.Queue())

    process = multiprocessing_ctx.Process(
        target=MultipleDataPipesToQueuesLoop, args=(datapipes, req_queues, res_queues, name)
    )
    return process, req_queues, res_queues
