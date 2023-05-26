# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

from itertools import zip_longest
from typing import Dict, List

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
]


class _RequestCounter:
    r"""
    _RequestCounter is used to synchronize between eventloops within the dispatching
    process. It guarantees to only handle the limit/pause/reset_epoch/resume request
    util all loops have received the same message.
    """
    exp_cnt: int
    _keys: List[str] = ["limit", "pause", "reset_epoch", "resume"]
    _cnt: Dict[str, int]
    _reached: Dict[str, bool]

    def __init__(self, exp_cnt: int):
        self.exp_cnt = exp_cnt
        self._cnt = {k: 0 for k in self._keys}
        self._reached = {k: False for k in self._keys}

    def increment(self, key: str) -> None:
        assert key in self._reached
        self._cnt[key] += 1
        assert self._cnt[key] <= self.exp_cnt
        if self._cnt[key] == self.exp_cnt:
            self._reached[key] = True

    def is_reached(self, key: str) -> bool:
        assert key in self._reached
        return self._reached[key]

    def reset(self, key: str) -> None:
        assert key in self._reached and self._reached[key]
        assert self._cnt[key] >= 1
        self._cnt[key] -= 1
        if self._cnt[key] == 0:
            self._reached[key] = False


def MultipleDataPipesToQueuesLoop(
    source_datapipes, req_queues, res_queues, process_name, worker_info, call_on_process_init=None, custom_reset_fn=None
):
    r"""
    Set the appropriate pipes and protocol server type, and create a loop over multiple datapipes
    with the protocol server in a non-blocking manner.

    Args:
        source_datapipe: DataPipe being iterated in the dispatching process
        req_queue: Multiprocessing queue providing requests from the worker process
        res_queue: Multiprocessing queue sending results to the worker process
        process_name: The name of process (used for logging and exception handling)
        worker_info: Worker information (worker id and number of workers)
        call_on_process_init: Not allowed by dispatching process for now.
        custom_reset_fn: Optional callable function to reset the DataPipe.
    """
    assert call_on_process_init is None, "``MultipleDataPipesToQueuesLoop`` does not support call_on_process_init"
    num_loops = len(source_datapipes)
    assert num_loops == len(req_queues) and num_loops == len(
        res_queues
    ), "``MultipleDataPipesToQueuesLoop`` requires the same number of datapipes, request queues and response queues"

    torch.set_num_threads(1)

    loops = []
    request_counter = _RequestCounter(num_loops)

    loop_id = 0
    for source_datapipe, req_queue, res_queue in zip(source_datapipes, req_queues, res_queues):
        loops.append(
            _create_datapipe_queue_loop(
                source_datapipe,
                req_queue,
                res_queue,
                process_name,
                loop_id,
                worker_info,
                custom_reset_fn,
                blocking_request_get=False,
                request_counter=request_counter,
            )
        )  # Non-blocking request with reset counters
        loop_id += 1

    # Using `zip_longest` to guarantee the process is terminated only when
    # all loops have received `TerminateRequest`
    for _ in zip_longest(*loops):
        # time.sleep to make Python switch context to get/send message in mp.Queue
        # TODO(ejguan): Microbenchmarked a synthetic non-replicable case that sleep perform similar to pass.
        #               A more comprehensive benchmarking in real-world scneario is needed.
        time.sleep(0)


def DataPipeToQueuesLoop(
    source_datapipe, req_queue, res_queue, process_name, worker_info, call_on_process_init=None, custom_reset_fn=None
):
    r"""
    Initialize with the given init function, set the appropriate pipe and protocol server type, and
    create a loop with the protocol server.

    Args:
        source_datapipe: DataPipe being iterated in the worker process
        req_queue: Multiprocessing queue providing requests from the main process
        res_queue: Multiprocessing queue sending results to the main process
        process_name: The name of process (used for logging and exception handling)
        worker_info: Worker information (worker id and number of workers)
        call_on_process_init: Callable function will be called at the time of worker process initialization.
            Users can provide it to modify the DataPipe grpah in the worker process.
        custom_reset_fn: Optional callable function to reset the DataPipe.
    """
    # Extract Serialization Wrapper
    source_datapipe = extract_wrapper(source_datapipe)

    if call_on_process_init is not None:
        source_datapipe = call_on_process_init(source_datapipe)

    torch.set_num_threads(1)

    loop = _create_datapipe_queue_loop(
        source_datapipe,
        req_queue,
        res_queue,
        process_name,
        worker_info.worker_id,
        worker_info,
        custom_reset_fn,
        blocking_request_get=True,
    )

    for _ in loop:
        pass


def _create_datapipe_queue_loop(
    source_datapipe,
    req_queue,
    res_queue,
    process_name,
    loop_id,
    worker_info,
    custom_reset_fn=None,
    blocking_request_get=True,
    request_counter=None,
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
        process_name=process_name,
        loop_id=loop_id,
        worker_info=worker_info,
        custom_reset_fn=custom_reset_fn,
        blocking_request_get=blocking_request_get,
        request_counter=request_counter,
    )


def CreateProcessForDataPipeline(
    multiprocessing_ctx, datapipe, process_name, worker_info, call_on_process_init=None, custom_reset_fn=None
):
    r"""
    Given a DataPipe, creates a new process with ``DataPipeToQueuesLoop`` as target,
    and returns ``(process, req_queue, res_queue)``.
    """
    req_queue = multiprocessing_ctx.Queue()
    res_queue = multiprocessing_ctx.Queue()
    process = multiprocessing_ctx.Process(
        target=DataPipeToQueuesLoop,
        args=(datapipe, req_queue, res_queue, process_name, worker_info, call_on_process_init, custom_reset_fn),
    )
    return process, req_queue, res_queue


def CreateProcessForMultipleDataPipelines(
    multiprocessing_ctx, datapipes, process_name, worker_info, custom_reset_fn=None
):
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
        target=MultipleDataPipesToQueuesLoop,
        args=(datapipes, req_queues, res_queues, process_name, worker_info, custom_reset_fn),
    )
    return process, req_queues, res_queues
