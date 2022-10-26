# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from queue import Empty as EmptyException

from torchdata.dataloader2 import communication


class Protocol:
    __slots__ = ("request_queue", "response_queue")

    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue


class ProtocolClient(Protocol):
    """
    ProtocolClient takes charge of putting requests into req_queue and returning results from res_queue.
    """

    _req_sent = None

    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._req_sent = None

    def can_take_request(self):
        return self._req_sent is None

    def waiting_for_response(self):
        return self._req_sent is not None

    def request_sent(self, request=True):
        if not self.can_take_request():
            raise Exception("Protocol only supports one request in the Queue")
        self._req_sent = request

    def request_served(self, result=None):
        if not self.waiting_for_response():
            raise Exception("Expected no peding requests, but something got served", result)
        self._req_sent = None

    def discard_existing_request(self):
        if self.waiting_for_response():
            response = self.response_queue.get(block=True)
            self.request_served(response)


class ProtocolServer(Protocol):
    """
    ProtocolServer takes charge of getting requests from req_queue and fetching data from source datapipe.
    """

    _req_received = None

    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._req_received = None

    def have_pending_request(self):
        return self._req_received is not None

    def get_new_request(self, block=False):
        if self.have_pending_request():
            raise Exception("Trying to get next request, while having one unserved")
        try:
            response = self.request_queue.get(block=block)
        except EmptyException:
            raise EmptyQueue("queue is empty")
        self._req_received = response
        return response
        # TODO(626): Validate supported requests

    def response_terminate(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        if not isinstance(self._req_received, communication.messages.TerminateRequest):
            raise Exception("Replaying with terminate status to other type of message")
        self.response_queue.put(communication.messages.TerminateResponse())
        self._req_received = None

    def response_reset_epoch(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        if not isinstance(self._req_received, communication.messages.ResetEpochRequest):
            raise Exception("Replaying with reset epoch status to other type of message")
        self.response_queue.put(communication.messages.ResetEpochResponse())
        self._req_received = None


class MapDataPipeQueueProtocolServer(ProtocolServer):
    def response_item(self, key, value):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.GetItemResponse(key, value))
        self._req_received = None

    def response_len(self, size):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.LenResponse(size))
        self._req_received = None

    def response_index_out_of_bound(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.StopIterationResponse())
        self._req_received = None


class MapDataPipeQueueProtocolClient(ProtocolClient):
    def request_len(self):
        if not self.can_take_request():
            raise Exception("Can not request len while we are still waiting response for previous request")
        request = communication.messages.LenRequest()
        self.request_queue.put(request)
        self.request_sent(request)

    def request_reset_epoch(self, *args):
        if not self.can_take_request():
            raise Exception("Can not reset while we are still waiting response for previous request")
        request = communication.messages.ResetEpochRequest(args)
        self.request_queue.put(request)
        self.request_sent(request)

    def request_item(self, index):
        if not self.can_take_request():
            raise Exception("Can not request item while we are still waiting response for previous request")
        request = communication.messages.GetItemRequest(index)
        self.request_queue.put(request)
        self.request_sent(request)

    def get_response_len(self, block=False, timeout=None):
        if not self.waiting_for_response():
            raise Exception("Can not expect any response without submitted request")
        try:
            response = self.response_queue.get(block=block, timeout=timeout)
        except TimeoutError:
            raise EmptyQueue("queue is empty")
        self.request_served(response)
        if not isinstance(response, communication.messages.LenResponse):
            raise Exception("Invalid response received")
        return response

    def get_response_item(self, block=False, timeout=None):
        if not self.waiting_for_response():
            raise Exception("Can not expect any response without submitted request")
        try:
            response = self.response_queue.get(block=block, timeout=timeout)
        except TimeoutError:
            raise EmptyQueue("queue is empty")
        self.request_served(response)
        # if not isinstance(response, communication.messages.GetItemResponse):
        #     raise Exception('Invalid response received')
        return response


class EmptyQueue(Exception):
    pass


class IterDataPipeQueueProtocolServer(ProtocolServer):
    def response_reset_iterator(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        if not isinstance(self._req_received, communication.messages.ResetIteratorRequest):
            raise Exception("Replaying with reset status to other type of message")
        self.response_queue.put(communication.messages.ResetIteratorResponse())
        self._req_received = None

    def response_next(self, value):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.GetNextResponse(value))
        self._req_received = None

    def response_stop_iteration(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.StopIterationResponse())
        self._req_received = None

    def response_invalid_state(self):
        if not self.have_pending_request():
            raise Exception("Attempting to reply with pending request")
        self.response_queue.put(communication.messages.InvalidStateResponse())
        self._req_received = None


class IterDataPipeQueueProtocolClient(ProtocolClient):
    def request_reset_iterator(self):
        if not self.can_take_request():
            raise Exception("Can not reset while we are still waiting response for previous request")
        request = communication.messages.ResetIteratorRequest()
        self.request_queue.put(request)
        self.request_sent(request)

    def request_reset_epoch(self, *args):
        if not self.can_take_request():
            raise Exception("Can not reset while we are still waiting response for previous request")
        request = communication.messages.ResetEpochRequest(args)
        self.request_queue.put(request)
        self.request_sent(request)

    def request_next(self):
        if not self.can_take_request():
            raise Exception("Can not request next item while we are still waiting response for previous request")
        request = communication.messages.GetNextRequest()
        self.request_queue.put(request)
        self.request_sent(request)

    def get_response_reset_iterator(self, block=False):
        try:
            response = self.response_queue.get(block=block)
        except EmptyException:
            raise EmptyQueue("queue is empty")
        self.request_served(response)

        if not isinstance(response, communication.messages.ResetIteratorResponse):
            raise Exception("Invalid response received")

    def get_response_next(self, block=False, timeout=None):
        if not self.waiting_for_response():
            raise Exception("Can not expect any response without submitted request")
        try:
            response = self.response_queue.get(block=block, timeout=timeout)
        except EmptyException:
            raise EmptyQueue("queue is empty")
        self.request_served(response)

        # TODO(629): Add possible response types validation here
        return response
