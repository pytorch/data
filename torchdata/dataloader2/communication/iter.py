# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import types
import warnings

from collections import deque
from functools import partial
from typing import Callable, Deque, List

from torch.utils.data import IterDataPipe
from torchdata._utils import ExceptionWrapper
from torchdata.dataloader2 import communication
from torchdata.dataloader2.graph import DataPipe, find_dps, list_dps, traverse_dps
from torchdata.dataloader2.random import SeedGenerator
from torchdata.dataloader2.utils import WorkerInfo


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
            except StopIteration:
                raise StopIteration
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


def DataPipeBehindQueues(
    source_datapipe, protocol, process_name, blocking_request_get=False, reset_iterator_counter=None
):
    """
    Indefinitely iterates over ``req_queue`` and passing values from source_datapipe to ``res_queue``.

    Request Types:
        `ResetEpoch` - Call the `reset_epoch_fn` on the protocol's DataPipe
        `ResetIterator` - Reset the iterator by calling `QueueWrapper`'s `reset_iterator` method
        `Terminate` - exits the infinite while loop
        `GetNext` - returns the value from the DataPipe, and handles exceptions such as `StopIteration` as appropriate

    Args:
        source_datapipe: DataPipe
        protocol: ``IterDataPipeQueueProtocolServer`` that contains ``req_queue`` and ``res_queue``
        process_name: Process name
        blocking_request_get: determines if ``protocol.get_new_request`` will block
        reset_iterator_counter: Optional counter to synchronize all loops that have received
            `ResetIteratorRequest` within the dispatching process. It would guarantee that
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

        if isinstance(request, communication.messages.ResetEpochRequest):
            source_datapipe = request.reset_fn(source_datapipe)
            protocol.response_reset_epoch()

        elif isinstance(request, communication.messages.ResetIteratorRequest):
            # Ensure only reset iterator once for the dispatching process
            if reset_iterator_counter is not None:
                reset_iterator_counter.increment()
                while not reset_iterator_counter.is_reached():
                    yield True
                # Sync between loops within the dispatching process
                source_datapipe.reset_iterator()
                yield True
                reset_iterator_counter.reset()
            source_datapipe.reset_iterator()
            protocol.response_reset_iterator()

        elif isinstance(request, communication.messages.PauseRequest):
            dp_list = list_dps(traverse_dps(source_datapipe))
            for dp in dp_list:
                # TODO: Remove this condition after there is `pause` support for round-robin sharding
                if isinstance(dp, _IterateQueueDataPipes):
                    warnings.warn("There is no support for `pause` with round-robin sharding at the moment.")
                elif hasattr(dp, "pause") and callable(dp.pause):
                    dp.pause()

            protocol.response_pause()
            yield True  # Return control

        elif isinstance(request, communication.messages.ResumeRequest):
            dp_list = list_dps(traverse_dps(source_datapipe))
            for dp in reversed(dp_list):
                # TODO: Remove this condition after there is `resume` support for round-robin sharding
                if isinstance(dp, _IterateQueueDataPipes):
                    raise RuntimeError("There is no support for `resume` with round-robin sharding at the moment.")
                elif hasattr(dp, "resume") and callable(dp.resume):
                    dp.resume()
            protocol.response_resume()
            yield True  # Return control

        elif isinstance(request, communication.messages.TerminateRequest):
            forever = False
            dispatch_dps = find_dps(traverse_dps(source_datapipe), _IterateQueueDataPipes)
            for dispatch_dp in dispatch_dps:
                dispatch_dp.request_terminate()

            protocol.response_terminate()

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
                    exc = ExceptionWrapper(where=f"in {process_name}")
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

    def reset_iterator(self):
        self._stop_iteration = False
        self.counter = 0
        self.protocol.request_reset_iterator()
        while True:
            try:
                self.protocol.get_response_reset_iterator()
                break
            except communication.protocol.EmptyQueue:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

    def pause(self):
        self.protocol.request_pause()
        while True:
            try:
                self.protocol.get_response_pause()
                break
            except communication.protocol.EmptyQueue:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

    def resume(self):
        self.protocol.request_resume()
        while True:
            try:
                self.protocol.get_response_resume()
                break
            except communication.protocol.EmptyQueue:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

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
        self.res_buffers: List[Deque] = [deque() for _ in range(len(datapipes))]
        self._terminated: bool = False

    def __iter__(self):
        total_pipes = len(self.datapipes)
        disabled_pipe = [False] * len(self.datapipes)
        cnt_disabled_pipes = 0

        for idx in range(total_pipes):
            self.datapipes[idx].protocol.request_next()

        while cnt_disabled_pipes < total_pipes:
            for idx in range(total_pipes):
                if not disabled_pipe[idx]:
                    # Check if buffer of the DataPipe is empty, if not, yield one before requesting next
                    if len(self.res_buffers[idx]):
                        response = self.res_buffers[idx].popleft()
                    else:
                        while not self._terminated:
                            try:
                                # Using non-blocking next to make sure termination reached
                                response = self.datapipes[idx].protocol.get_response_next(block=False)
                                break
                            except communication.protocol.EmptyQueue:
                                time.sleep(DEFAULT_NON_BLOCKING_SLEEP)
                    if isinstance(response, communication.messages.StopIterationResponse):
                        disabled_pipe[idx] = True
                        cnt_disabled_pipes += 1
                        continue
                    if isinstance(response, communication.messages.InvalidStateResponse):
                        raise communication.iter.InvalidStateResetRequired
                    if isinstance(response, communication.messages.TerminateResponse):
                        raise communication.iter.TerminateRequired
                    if isinstance(response, communication.messages.WorkerExceptionResponse):
                        response.exc.reraise()
                    if len(self.res_buffers[idx]) == 0:  # Only request if buffer is empty
                        self.datapipes[idx].protocol.request_next()
                    yield response.value

    def reset(self):
        # NonBlocking DataPipes do not reset automatically, have to do it manually
        for dp in self.datapipes:
            dp.reset_iterator()

    def reset_epoch(
        self, reset_fn: Callable[[WorkerInfo, SeedGenerator, DataPipe], DataPipe], seed_generator: SeedGenerator
    ):
        for dp in self.datapipes:
            dp.protocol.discard_existing_request()
        num_workers = len(self.datapipes)
        for worker_id, dp in enumerate(self.datapipes):
            worker_info = WorkerInfo(num_workers, worker_id)
            worker_seed_generator = seed_generator.spawn(worker_id)
            dp.protocol.request_reset_epoch(
                partial(reset_fn, worker_info=worker_info, seed_generator=worker_seed_generator)
            )
        for dp in self.datapipes:
            while True:
                try:
                    dp.protocol.get_response_reset_epoch()
                    break
                except communication.protocol.EmptyQueue:
                    if NonBlocking.not_available_hook is not None:
                        NonBlocking.not_available_hook()

    def request_pause(self):
        # Store results of pending requests
        for idx, dp in enumerate(self.datapipes):
            if dp.protocol.waiting_for_response():
                res = dp.protocol.get_response_next(block=True)
                self.res_buffers[idx].append(res)
        for dp in self.datapipes:
            dp.pause()

    def request_resume(self):
        for dp in self.datapipes:
            dp.resume()

    def request_terminate(self):
        self._terminated = True
        for dp in self.datapipes:
            dp.protocol.request_terminate()
