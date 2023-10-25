# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import types
import warnings

from collections import deque
from itertools import cycle
from typing import Callable, Deque, List, Optional

from torch.utils.data import IterDataPipe
from torchdata._utils import ExceptionWrapper
from torchdata.dataloader2 import communication
from torchdata.dataloader2.graph import DataPipe, find_dps, list_dps, traverse_dps
from torchdata.dataloader2.random import SeedGenerator
from torchdata.dataloader2.utils import process_reset_fn


DEFAULT_NON_BLOCKING_SLEEP = 0.001

__all__ = [
    "DataPipeBehindQueues",
    "EnsureNonBlockingDataPipe",
    "InvalidStateResetRequired",
    "NonBlocking",
    "NotAvailable",
    "QueueWrapper",
    "default_not_available_hook",
]


def default_not_available_hook():
    time.sleep(DEFAULT_NON_BLOCKING_SLEEP)


class NotAvailable(Exception):
    pass


class InvalidStateResetRequired(Exception):
    """
    Returned by DataPipe when it is expecting to get reset request,
    for example RouterDataPipe expecting all workers to request reset.
    """

    pass


class TerminateRequired(Exception):
    """
    Returned by DataPipe when it is expecting to get terminate request,
    for example it got terminate request from other source and at the process
    of stopping.
    """

    pass


class NonBlocking(IterDataPipe):
    not_available_hook = default_not_available_hook

    def __iter__(self):
        self.reset_iterator()
        return self

    def __next__(self):
        while True:
            try:
                return self.nonblocking_next()
            except NotAvailable:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

    def nonblocking_next(self):
        raise NotImplementedError("nonblocking_next is not implemented for %s" % self.__class__)

    def reset_iterator(self):
        raise NotImplementedError("reset_iterator is not implemented for %s" % self.__class__)

    @staticmethod
    def register_not_available_hook(hook_function):
        NonBlocking.not_available_hook = hook_function


def EnsureNonBlockingDataPipe(validated_datapipe):
    if not isinstance(validated_datapipe, IterDataPipe):
        raise Exception("Not Iterable DataPipe " + str(validated_datapipe.__class__))
    if isinstance(validated_datapipe, NonBlocking):
        return validated_datapipe
    if not hasattr(validated_datapipe, "_as_iterator"):
        validated_datapipe._as_iterator = None  # type: ignore[attr-defined]
    if not hasattr(validated_datapipe, "nonblocking_next"):

        def nonblocking_next(self):
            if self._as_iterator is None:
                self._as_iterator = iter(self)
            return next(self._as_iterator)

        validated_datapipe.nonblocking_next = types.MethodType(  # type: ignore[attr-defined]
            nonblocking_next, validated_datapipe
        )
    if not hasattr(validated_datapipe, "reset_iterator"):

        def reset_iterator(self):
            self._as_iterator = None

        validated_datapipe.reset_iterator = types.MethodType(  # type: ignore[attr-defined]
            reset_iterator, validated_datapipe
        )
    return validated_datapipe


def _sync_recv(request_counter, msg):
    if request_counter is not None:
        request_counter.increment(msg)
        # Make sure all loops have reached
        while not request_counter.is_reached(msg):
            yield True


def _sync_resp(request_counter, msg):
    if request_counter is not None:
        request_counter.reset(msg)
        while request_counter.is_reached(msg):
            yield True


def DataPipeBehindQueues(
    source_datapipe,
    protocol,
    process_name,
    loop_id,
    worker_info,
    custom_reset_fn,
    blocking_request_get=False,
    request_counter=None,
):
    """
    Indefinitely iterates over ``req_queue`` and passing values from source_datapipe to ``res_queue``.

    Request Types:
        `ResetEpoch` - Call the `reset_epoch_fn` on the protocol's DataPipe and reset DataPipe iterator
        `Terminate` - exits the infinite while loop
        `GetNext` - returns the value from the DataPipe, and handles exceptions such as `StopIteration` as appropriate
        `Limit` - Set limit to the DataPipe graph
        `Pause` - Pause
        the DataPipe graph
        `Resume` - Resume the DataPipe graph

    Args:
        source_datapipe: DataPipe
        protocol: ``IterDataPipeQueueProtocolServer`` that contains ``req_queue`` and ``res_queue``
        process_name: Process name
        loop_id: Loop ID
        worker_info: Worker info include worker id and number of workers
        custom_reset_fn: function to call after each request is received
        blocking_request_get: determines if ``protocol.get_new_request`` will block
        request_counter: Optional counter to synchronize all loops that have received requests for
            reset/limit/pause/resume within the dispatching process. It would guarantee that
            all loops starts to reset iterator and get next element at the same time.
    """
    if not isinstance(protocol, communication.protocol.IterDataPipeQueueProtocolServer):
        raise Exception("Expecting IterDataPipeQueueProtocolServer, got", protocol)
    source_datapipe = EnsureNonBlockingDataPipe(source_datapipe)
    forever = True
    while forever:
        try:
            # TODO: Non-blocking call is extremely slow here for python.mp, need to figure out a good workaround
            request = protocol.get_new_request(block=blocking_request_get)
        except communication.protocol.EmptyQueue:
            yield True
            continue

        # TODO: Handle Error caused by requests other than GetNext and send it to main process
        if isinstance(request, communication.messages.ResetEpochRequest):
            yield from _sync_recv(request_counter, "reset_epoch")
            distributed_shared_seed = request_counter is not None
            if request_counter is None or loop_id == 0:
                seed_generator = request.seed_generator
                iter_reset_fn = request.iter_reset_fn
                dispatching_dps = find_dps(traverse_dps(source_datapipe), _IterateQueueDataPipes)
                for dp in dispatching_dps:
                    dp.reset_epoch(seed_generator, iter_reset_fn)
                source_datapipe = process_reset_fn(
                    source_datapipe,
                    worker_info,
                    seed_generator,
                    distributed_shared_seed,
                    iter_reset_fn,
                    custom_reset_fn,
                )
            source_datapipe.reset_iterator()
            yield from _sync_resp(request_counter, "reset_epoch")
            protocol.response_reset_epoch()
            yield True  # Returns control

        elif isinstance(request, communication.messages.LimitRequest):
            yield from _sync_recv(request_counter, "limit")
            if request_counter is None or loop_id == 0:
                num_batches = request.num_batches
                limit_fn = request.limit_fn
                worker_num_batches = num_batches if request.worker_num_batches is None else request.worker_num_batches
                # Send limit to the worker/dispatching process
                dispatching_dps = find_dps(traverse_dps(source_datapipe), _IterateQueueDataPipes)
                for dp in dispatching_dps:
                    dp.request_limit(num_batches, limit_fn, worker_num_batches)
                if limit_fn is not None:
                    # Set limit to the DataPipe graph in worker/dispatching process
                    source_datapipe = limit_fn(source_datapipe, worker_num_batches)
            yield from _sync_resp(request_counter, "limit")
            protocol.response_limit()
            yield True  # Returns control

        elif isinstance(request, communication.messages.PauseRequest):
            yield from _sync_recv(request_counter, "pause")
            if request_counter is None or loop_id == 0:
                graph = traverse_dps(source_datapipe)
                dp_list = list_dps(graph)
                for dp in dp_list:
                    if hasattr(dp, "pause") and callable(dp.pause):
                        dp.pause()
                dispatching_dps = find_dps(graph, _IterateQueueDataPipes)
                for dp in dispatching_dps:
                    dp.request_pause(request.pause_fn)
                if request.pause_fn is not None:
                    source_datapipe = request.pause_fn(source_datapipe)
            yield from _sync_resp(request_counter, "pause")
            protocol.response_pause()
            yield True  # Returns control

        elif isinstance(request, communication.messages.ResumeRequest):
            yield from _sync_recv(request_counter, "resume")
            if request_counter is None or loop_id == 0:
                if request.resume_fn is not None:
                    source_datapipe = request.resume_fn(source_datapipe)
                graph = traverse_dps(source_datapipe)
                # Send resume to the dispatching process
                dispatching_dps = find_dps(graph, _IterateQueueDataPipes)
                for dp in dispatching_dps:
                    dp.request_resume(request.resume_fn)
                for dp in reversed(list_dps(graph)):
                    if hasattr(dp, "resume") and callable(dp.resume):
                        dp.resume()
            yield from _sync_resp(request_counter, "resume")
            protocol.response_resume()
            yield True  # Returns control

        elif isinstance(request, communication.messages.TerminateRequest):
            forever = False
            dispatch_dps = find_dps(traverse_dps(source_datapipe), _IterateQueueDataPipes)
            for dispatch_dp in dispatch_dps:
                dispatch_dp.request_terminate()
            protocol.response_terminate()
            yield True  # Returns control

        elif isinstance(request, communication.messages.GetNextRequest):
            while forever:
                if protocol.is_paused():
                    protocol.response_stop_iteration()
                    warnings.warn(
                        "Cannot `GetNext` after `Pause` has been called. "
                        "`Resume` must be called first before additional elements can be yielded."
                    )
                    yield True
                    break
                try:
                    value = source_datapipe.nonblocking_next()
                except NotAvailable:
                    yield True
                    continue
                except StopIteration:
                    protocol.response_stop_iteration()
                    yield True
                    break
                except InvalidStateResetRequired:
                    protocol.response_invalid_state()
                    yield True
                    break
                except Exception:
                    exc = ExceptionWrapper(where=f"in {process_name} {loop_id}")
                    protocol.response_worker_exception(exc)
                    return
                protocol.response_next(value)
                yield True  # Returns control
                break
        else:
            raise Exception("Unrecognized type of request received", request)


class QueueWrapper(NonBlocking):
    """
    Creates an IterDataPipe which sends requests and reads the response from the DataLoader.Queue.
    The input is a ProtocolClient that contains request queue and response queue.
    """

    def __init__(self, protocol, response_wait_time=0.00001):
        if not isinstance(protocol, communication.protocol.IterDataPipeQueueProtocolClient):
            raise Exception("Got", protocol)
        self.protocol = protocol
        self.counter = 0
        self._stop_iteration = False
        self._response_wait_time = response_wait_time

    def request_reset_epoch(self, seed_generator, iter_reset_fn):
        self._stop_iteration = False
        self.counter = 0
        self.protocol.request_reset_epoch(seed_generator, iter_reset_fn)

    def _get_response(self, fn_name) -> None:
        assert hasattr(self.protocol, fn_name) and callable(getattr(self.protocol, fn_name))
        get_response_fn = getattr(self.protocol, fn_name)
        while True:
            try:
                get_response_fn()
                break
            except communication.protocol.EmptyQueue:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

    def get_reset_epoch_response(self) -> None:
        self._get_response("get_response_reset_epoch")

    def request_limit(
        self,
        num_batches: Optional[int],
        limit_fn: Optional[Callable[[DataPipe, Optional[int]], DataPipe]] = None,
        worker_num_batches: Optional[int] = None,
    ) -> None:
        self.protocol.request_limit(num_batches, limit_fn, worker_num_batches)

    def get_limit_response(self) -> None:
        self._get_response("get_response_limit")

    def request_pause(self, pause_fn: Optional[Callable[[DataPipe], DataPipe]] = None) -> None:
        self.protocol.request_pause(pause_fn)

    def get_pause_response(self) -> None:
        self._get_response("get_response_pause")

    def request_resume(self, resume_fn: Optional[Callable[[DataPipe], DataPipe]] = None) -> None:
        self.protocol.request_resume(resume_fn)

    def get_resume_response(self) -> None:
        self._get_response("get_response_resume")

    def nonblocking_next(self):
        if self._stop_iteration:
            raise Exception("`next` or `nonblocking_next` called after receiving StopIteration")
        if self.protocol.can_take_request():
            self.protocol.request_next()
        try:
            response = self.protocol.get_response_next(block=True, timeout=self._response_wait_time)
        except communication.protocol.EmptyQueue:
            raise NotAvailable
        if isinstance(response, communication.messages.StopIterationResponse):
            self._stop_iteration = True
            raise StopIteration
        if isinstance(response, communication.messages.InvalidStateResponse):
            raise NotAvailable
        return response.value


class _IterateQueueDataPipes(IterDataPipe):
    r"""
    Takes in ``QueueWrapper``s and iterates through them in a round-robin manner to get batches one-by-one.

    Typically, each worker has one ``QueueWrapper``.
    """

    def __init__(self, datapipes):
        # TODO(VitalyFedyunin): Consider combining _IterateQueueDataPipes and QueueWrapper
        #                       into one class, which supports any number of queues.
        for dp in datapipes:
            if not isinstance(dp, communication.iter.QueueWrapper):
                raise Exception("Source datapipes should be an instance of iter.QueueWrapper")
        self.datapipes = datapipes
        self._num_processes = len(datapipes)
        self.res_buffers: List[Deque] = [deque() for _ in range(len(datapipes))]
        self._terminated: bool = False
        self._limit: Optional[int] = None
        self._request_cnt: int = 0

    def __iter__(self):
        disabled_pipe = [False] * len(self.datapipes)
        cnt_disabled_pipes = 0

        total_req_cnt = 0
        req_idx_cycle = cycle(range(self._num_processes))
        req_idx = next(req_idx_cycle)
        total_res_cnt = 0
        res_idx_cycle = cycle(range(self._num_processes))
        res_idx = next(res_idx_cycle)

        while cnt_disabled_pipes < self._num_processes and not self._terminated:
            # Send a round of requests until limit is reached (limit is smaller than total pipes)
            for _ in range(self._num_processes):
                if not disabled_pipe[req_idx]:
                    self.datapipes[req_idx].protocol.request_next()
                    self._request_cnt += 1
                total_req_cnt += 1
                req_idx = next(req_idx_cycle)
                if self._limit is not None and self._request_cnt == self._limit:
                    break
            # Receive responses from each of the workers with pending requests
            while total_res_cnt < total_req_cnt and cnt_disabled_pipes < self._num_processes:
                disabled = disabled_pipe[res_idx]
                if not disabled:
                    if len(self.res_buffers[res_idx]):
                        response = self.res_buffers[res_idx].popleft()
                    else:
                        while not self._terminated:
                            try:
                                # Using non-blocking next to make sure termination reached
                                response = self.datapipes[res_idx].protocol.get_response_next(block=False)
                                break
                            except communication.protocol.EmptyQueue:
                                time.sleep(DEFAULT_NON_BLOCKING_SLEEP)
                    if isinstance(response, communication.messages.InvalidStateResponse):
                        raise communication.iter.InvalidStateResetRequired
                    if isinstance(response, communication.messages.TerminateResponse):
                        raise communication.iter.TerminateRequired
                    if isinstance(response, communication.messages.WorkerExceptionResponse):
                        response.exc.reraise()
                    if self._terminated:
                        break
                    if isinstance(response, communication.messages.StopIterationResponse):
                        disabled_pipe[res_idx] = True
                        cnt_disabled_pipes += 1
                        disabled = True
                        req_idx = next(req_idx_cycle)
                    else:
                        # Only request if buffer is empty and has not reached the limit
                        if len(self.res_buffers[res_idx]) == 0 and (
                            self._limit is None or self._request_cnt < self._limit
                        ):
                            self.datapipes[req_idx].protocol.request_next()
                            req_idx = next(req_idx_cycle)
                            self._request_cnt += 1
                            total_req_cnt += 1
                    total_res_cnt += 1
                res_idx = next(res_idx_cycle)
                if not disabled:
                    yield response.value

    def reset_epoch(
        self,
        seed_generator: SeedGenerator,
        iter_reset_fn: Optional[Callable[[DataPipe], DataPipe]],
    ):
        self._request_cnt = 0
        for dp in self.datapipes:
            dp.protocol.discard_existing_request()
        for worker_id, dp in enumerate(self.datapipes):
            worker_seed_generator = seed_generator.spawn(worker_id)
            dp.request_reset_epoch(worker_seed_generator, iter_reset_fn)
        for dp in self.datapipes:
            dp.get_reset_epoch_response()

    def request_pause(self, pause_fn: Optional[Callable[[DataPipe], DataPipe]] = None) -> None:
        # Store results of pending requests
        for idx, dp in enumerate(self.datapipes):
            if dp.protocol.waiting_for_response():
                res = dp.protocol.get_response_next(block=True)
                self.res_buffers[idx].append(res)
        for dp in self.datapipes:
            dp.request_pause(pause_fn)
        for dp in self.datapipes:
            dp.get_pause_response()

    def request_resume(self, resume_fn: Optional[Callable[[DataPipe], DataPipe]] = None) -> None:
        for dp in self.datapipes:
            dp.request_resume(resume_fn)
        for dp in self.datapipes:
            dp.get_resume_response()
        self._request_cnt = 0

    def request_limit(
        self,
        num_batches: Optional[int],
        limit_fn: Optional[Callable[[DataPipe, Optional[int]], DataPipe]] = None,
        worker_num_batches: Optional[int] = None,
    ) -> None:
        self._limit = num_batches if worker_num_batches is None else worker_num_batches
        avg_num_batches = num_batches if num_batches is None else num_batches // self._num_processes
        batch_remainder = 0 if num_batches is None else num_batches % self._num_processes
        for idx, dp in enumerate(self.datapipes):
            ext_batch = 1 if batch_remainder > idx else 0
            wnb = None if avg_num_batches is None or worker_num_batches is not None else avg_num_batches + ext_batch
            dp.request_limit(num_batches, limit_fn, wnb)
        for dp in self.datapipes:
            dp.get_limit_response()

    def request_terminate(self):
        self._terminated = True
        for dp in self.datapipes:
            dp.protocol.discard_existing_request()
        for dp in self.datapipes:
            dp.protocol.request_terminate()
