# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata._utils import ExceptionWrapper


class DataLoaderQueueMessage:
    pass


class Request(DataLoaderQueueMessage):
    pass


class Response(DataLoaderQueueMessage):
    pass


class ResetEpochRequest(Request):
    __slots__ = ("seed_generator", "iter_reset_fn")

    def __init__(self, seed_generator, iter_reset_fn):
        self.seed_generator = seed_generator
        self.iter_reset_fn = iter_reset_fn


class ResetEpochResponse(Response):
    pass


class LimitRequest(Request):
    __slots__ = ("num_batches", "limit_fn", "worker_num_batches")

    def __init__(self, num_batches, limit_fn, worker_num_batches=None):
        self.num_batches = num_batches
        self.limit_fn = limit_fn
        self.worker_num_batches = worker_num_batches


class LimitResponse(Response):
    pass


class PauseRequest(Request):
    __slots__ = "pause_fn"

    def __init__(self, pause_fn):
        self.pause_fn = pause_fn


class PauseResponse(Response):
    pass


class ResumeRequest(Request):
    __slots__ = "resume_fn"

    def __init__(self, resume_fn):
        self.resume_fn = resume_fn


class ResumeResponse(Response):
    pass


class TerminateRequest(Request):
    pass


class TerminateResponse(Response):
    pass


class LenRequest(Request):
    pass


class LenResponse(Response):
    __slots__ = "len"

    def __init__(self, len):
        self.len = len


class GetItemRequest(Request):
    __slots__ = "key"

    def __init__(self, key):
        self.key = key


class GetItemResponse(Response):
    __slots__ = ("key", "value")

    def __init__(self, key, value):
        self.key = key
        self.value = value


class GetNextRequest(Request):
    pass


class GetNextResponse(Response):
    __slots__ = "value"

    def __init__(self, value):
        self.value = value


class StopIterationResponse(Response):
    pass


class InvalidStateResponse(Response):
    """
    Returned by DataPipe when it is expecting to get reset request,
    for example RouterDataPipe expecting all workers to request reset'
    """

    pass


class WorkerExceptionResponse(Response):
    __slots__ = "exc"

    def __init__(self, exc: ExceptionWrapper):
        self.exc: ExceptionWrapper = exc
