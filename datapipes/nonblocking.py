class DataLoaderQueueMessage(object):
    pass


class Request(DataLoaderQueueMessage):
    pass


class Response(DataLoaderQueueMessage):
    pass


class NotAvailable(Exception):
    pass


class ResetIteratorRequest(Request):
    pass


class ResetIteratorResponse(Response):
    pass


class StopIteratorRequest(Request):
    pass


class StopIteratorResponse(Response):
    pass


class LenRequest(Request):
    pass


class LenResponse(Response):
    def __init__(self, len):
        self.len = len


class GetItemRequest(Request):
    def __init__(self, key):
        self.key = key


class GetItemResponse(Response):
    def __init__(self, key, value):
        self.key = key
        self.value = value


class GetNextRequest(Request):
    pass


class GetNextResponse(Response):
    def __init__(self, value):
        self.value = value
