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


TIME_SLEEP_BETWEEN_CHECKING_DIFFERENT_QUEUES = 0.00000001


def MultipleDataPipesToQueuesLoop(source_datapipes, req_queues, res_queues, call_on_process_init=None):
    r"""
    Set the appropriate pipes and protocol server type, and create a loop over multiple datapipes
    with the protocol server in a non-blocking manner.
    """
    assert call_on_process_init is None, "``MultipleDataPipesToQueuesLoop`` does not support call_on_process_init"
    assert len(source_datapipes) == len(req_queues) and len(req_queues) == len(
        res_queues
    ), "``MultipleDataPipesToQueuesLoop`` requires the same number of datapipes, request queues and response queues"

    torch.set_num_threads(1)

    loops = []

    for source_datapipe, req_queue, res_queue in zip(source_datapipes, req_queues, res_queues):
        loops.append(
            _create_datapipe_queue_loop(source_datapipe, req_queue, res_queue, blocking_request_get=False)
        )  # Non-blocking request

    # Using `zip_longest` to guarantee the process is terminated only when
    # all loops have received `TerminateRequest`
    for _ in zip_longest(*loops):
        # TODO(ejguan): Check python MP implementation why this sleep impacts queues statuses
        # This magical sleep allows mp queue messages to travel faster
        time.sleep(TIME_SLEEP_BETWEEN_CHECKING_DIFFERENT_QUEUES)
        pass


def DataPipeToQueuesLoop(source_datapipe, req_queue, res_queue, call_on_process_init=None):
    r"""
    Initialize with the given init function, set the appropriate pipe and protocol server type, and
    create a loop with the protocol server.
    """
    if call_on_process_init is not None:
        call_on_process_init(source_datapipe)

    torch.set_num_threads(1)

    loop = _create_datapipe_queue_loop(source_datapipe, req_queue, res_queue, blocking_request_get=True)

    for _ in loop:
        pass


def _create_datapipe_queue_loop(source_datapipe, req_queue, res_queue, blocking_request_get=True):
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
        blocking_request_get=True,
    )


def CreateProcessForDataPipeline(multiprocessing_ctx, datapipe, call_on_process_init=None):
    r"""
    Given a DataPipe, creates a new process with ``DataPipeToQueuesLoop`` as target,
    and returns ``(process, req_queue, res_queue)``.
    """
    req_queue = multiprocessing_ctx.Queue()
    res_queue = multiprocessing_ctx.Queue()
    process = multiprocessing_ctx.Process(
        target=DataPipeToQueuesLoop, args=(datapipe, req_queue, res_queue, call_on_process_init)
    )
    return process, req_queue, res_queue


def CreateThreadForDataPipeline(datapipe):
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

    process = threading.Thread(target=DataPipeToQueuesLoop, args=(new_datapipe, req_queue, res_queue), daemon=True)
    return process, req_queue, res_queue, new_datapipe


def CreateProcessForMultipleDataPipelines(multiprocessing_ctx, datapipes):
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
        target=MultipleDataPipesToQueuesLoop, args=(datapipes, req_queues, res_queues)
    )
    return process, req_queues, res_queues
