# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import threading
import time

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
    "SpawnProcessForDataPipeline",
    "SpawnThreadForDataPipeline",
]

TIME_SLEEP_BETWEEN_CHECKING_DIFFERENT_QUEUES = 0.00000001

# TODO(VitalyFedyunin): Find better names to the two functions below as they are separate thread/process/work-items
# TODO(VitalyFedyunin): Can combine Multiple and Single functions by checking size of pipes_and_queues and deciding block/non-block.


def MultipleDataPipesToQueuesLoop(pipes_and_queues, call_locally_fn=None):
    if call_locally_fn is not None:
        raise Exception("MultipleDataPipesToQueuesLoop does not support call_locally_fn")
    torch.set_num_threads(1)

    resets_counter = [0]

    iterators = []
    for source_datapipe, req_queue, res_queue in pipes_and_queues:
        iterators.append(
            DataPipeToQueuesLoopIterator(
                source_datapipe,
                req_queue,
                res_queue,
                blocking_request_get=False,
                resets_to_proceed=len(pipes_and_queues),
                resets_counter=resets_counter,
            )
        )

    # TODO(VitalyFedyunin): Maybe better way to combine iterators
    for _ in zip(*iterators):
        # TODO(VitalyFedyunin): Check python MP implementation why this sleep impacts queues statuses
        # This magical sleep allows mp queue messages to travel faster
        time.sleep(TIME_SLEEP_BETWEEN_CHECKING_DIFFERENT_QUEUES)
        pass


def DataPipeToQueuesLoop(source_datapipe, req_queue, res_queue, call_locally_fn=None):
    if call_locally_fn is not None:
        source_datapipe = call_locally_fn(source_datapipe)
    torch.set_num_threads(1)
    for _ in DataPipeToQueuesLoopIterator(source_datapipe, req_queue, res_queue, blocking_request_get=True):
        pass


def DataPipeToQueuesLoopIterator(
    source_datapipe, req_queue, res_queue, blocking_request_get=True, resets_to_proceed=1, resets_counter=[]
):
    if isinstance(source_datapipe, IterDataPipe):
        pipe_type = communication.iter
        protocol_type = communication.protocol.IterDataPipeQueueProtocolServer
    elif isinstance(source_datapipe, MapDataPipe):
        pipe_type = communication.map  # type: ignore[misc]
        protocol_type = communication.protocol.MapDataPipeQueueProtocolServer  # type: ignore[assignment]
    else:
        raise Exception("Only supports IterDataPipe or MapDataPipe, got", source_datapipe)

    for _ in pipe_type.DataPipeBehindQueues(
        source_datapipe,
        protocol_type(req_queue, res_queue),
        blocking_request_get=blocking_request_get,
        resets_to_proceed=resets_to_proceed,
        resets_counter=resets_counter,
    ):
        yield True


def SpawnProcessForDataPipeline(multiprocessing_ctx, datapipe, call_locally_fn=None):
    req_queue = multiprocessing_ctx.Queue()
    res_queue = multiprocessing_ctx.Queue()
    process = multiprocessing_ctx.Process(
        target=DataPipeToQueuesLoop, args=(datapipe, req_queue, res_queue, call_locally_fn)
    )
    return process, req_queue, res_queue


def SpawnProcessForMultipleDataPipelines(multiprocessing_ctx, datapipes, call_locally_fn=None):
    pipes_and_queues = []
    for dp in datapipes:
        req_queue = multiprocessing_ctx.Queue()
        res_queue = multiprocessing_ctx.Queue()
        pipes_and_queues.append((dp, req_queue, res_queue))

    process = multiprocessing_ctx.Process(
        target=MultipleDataPipesToQueuesLoop, args=(pipes_and_queues, call_locally_fn)
    )
    return process, pipes_and_queues


def SpawnThreadForDataPipeline(datapipe):
    r"""
    Given a DataPipe, creates a copy of the DataPipe, starts a new Thread with DataPipeToQueuesLoop as target,
    and return the process, req_queue, res_queue, thread_local_datapipe.
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
