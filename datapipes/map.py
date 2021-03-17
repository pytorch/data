import time
import types

from torch.utils.data import Dataset, IterDataPipe, IterableDataset, functional_datapipe, non_deterministic

import datapipes
import datapipes.nonblocking as nonblocking

DEFAULT_NON_BLOCKING_SLEEP = 0.001


def MapDataPipe(Dataset):
    pass


def default_not_available_hook():
    time.sleep(DEFAULT_NON_BLOCKING_SLEEP)


class NonBlocking(IterDataPipe):
    not_available_hook = default_not_available_hook

    def __getitem__(self, key):
        while True:
            try:
                return self.nonblocking_get(key)
            except nonblocking.NotAvailable:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

    def __len__(self):
        while True:
            try:
                return self.nonblocking_len()
            except nonblocking.NotAvailable:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

    def nonblocking_get(self, key):
        raise NotImplementedError(
            "nonblocking_get is not implemented for %s" % self.__class__)

    def nonblocking_len(self):
        raise NotImplementedError(
            "nonblocking_len is not implemented for %s" % self.__class__)

    def reset_iterator(self):
        raise NotImplementedError(
            "reset_iterator is not implemented for %s" % self.__class__)

    @staticmethod
    def register_not_available_hook(hook_function):
        NonBlocking.not_available_hook = hook_function


def EnsureNonBlockingDataPipe(validated_datapipe):
    if not isinstance(validated_datapipe, Dataset):
        raise Exception('Not instance of DataPipe ' +
                        str(validated_datapipe.__class__))

    # TODO(VitalyFedyunin): Check if not IterDataPipe

    if isinstance(validated_datapipe, NonBlocking):
        return validated_datapipe

    if not hasattr(validated_datapipe, '__getitem__'):
        raise Exception('Missing __getitem__ in ', str(validated_datapipe.__class__))

    if not hasattr(validated_datapipe, 'nonblocking_get'):
        def nonblocking_get(self, key):
            return self[key]
        setattr(validated_datapipe, 'nonblocking_get', nonblocking_get)
        validated_datapipe.nonblocking_get = types.MethodType(
            nonblocking_get, validated_datapipe)

    if not hasattr(validated_datapipe, 'nonblocking_len'):
        def nonblocking_len(self):
            return self.__len__()
        setattr(validated_datapipe, 'nonblocking_len', nonblocking_len)
        validated_datapipe.nonblocking_len = types.MethodType(
            nonblocking_len, validated_datapipe)

    if not hasattr(validated_datapipe, 'reset_iterator'):
        def reset_iterator(self):
            pass
        setattr(validated_datapipe, 'reset_iterator', reset_iterator)
        validated_datapipe.reset_iterator = types.MethodType(
            reset_iterator, validated_datapipe)

    return validated_datapipe


# Creates iter.DataPipe which reads data from the DataLoader.Queue
class QueueWrapper(NonBlocking):
    def __init__(self, protocol, response_wait_time=0.00001):
        self._req_q = protocol.request_queue
        self._res_q = protocol.response_queue
        self._req_sent = False
        self._req_key = None
        self._stop_iteration = False
        self._response_wait_time = response_wait_time

    def reset_iterator(self):
        if self._req_sent:
            raise Exception(
                'Can not reset QueueWrapper while it is still waiting for response for', self._req_q.name)
        self._stop_iteration = False
        self._req_q.put(datapipes.nonblocking.ResetIteratorRequest())
        while True:
            try:
                value = self._res_q.get(block=False)
                break
            except:
                if NonBlocking.not_available_hook is not None:
                    NonBlocking.not_available_hook()

        if not isinstance(value, datapipes.nonblocking.ResetIteratorResponse):
            raise Exception('Invalid response received')

    def nonblocking_get(self, key):
        if self._stop_iteration:
            raise Exception(
                '`__hetitem__` or `nonblocking_get` called after receiving StopIteration')
        if not self._req_sent:
            self._req_q.put(datapipes.nonblocking.GetItemRequest(key))
            self._req_sent = True
            self._req_key = key
            # return control to eventloop to fill results if possible
            # eventloop.EventLoop.iteration()
            # return control as previous solution ends with infinite loop
            # can be only used in case of event loop
            # raise datapipes.nonblocking.NotAvailable
        try:
            response = self._res_q.get(
                block=True, timeout=self._response_wait_time)
        except:  # TODO: Catch only timeout exceptions
            raise nonblocking.NotAvailable
        self._req_sent = False
        if isinstance(response, StopIteration):
            self._stop_iteration = True
            raise StopIteration

        if not isinstance(response, nonblocking.GetItemResponse):
            raise Exception('Expected GetItemResponse got', response)

        if response.key != self._req_key:
            raise Exception('Got response with wrong key', response.key, self._req_key)
        return response.value

    def nonblocking_len(self):
        if self._stop_iteration:
            raise Exception(
                '`__len__` or `nonblocking_len` called after receiving StopIteration')
        if not self._req_sent:
            self._req_q.put(nonblocking.LenRequest())
            self._req_sent = True
            # return control to eventloop to fill results if possible
            # eventloop.EventLoop.iteration()
            # return control as previous solution ends with infinite loop
            # can be only used in case of event loop
            # raise datapipes.nonblocking.NotAvailable
        try:
            response = self._res_q.get(
                block=True, timeout=self._response_wait_time)
        except:  # TODO: Catch only timeout exceptions
            raise nonblocking.NotAvailable
        self._req_sent = False
        if isinstance(response, StopIteration):
            self._stop_iteration = True
            raise StopIteration

        if not isinstance(response, nonblocking.LenResponse):
            raise Exception('Got response to the wrong query', response)

        return response.len

# Indefinitely iterates over req_queue and passing values from source_datapipe to res_queue
# If raise_stop is true, raises exception when StopIteration received from the source_datapipe


def DataPipeBehindQueues(source_datapipe, protocol, full_stop=False, blocking_request_get=False):
    req_queue = protocol.request_queue
    res_queue = protocol.response_queue
    
    source_datapipe = EnsureNonBlockingDataPipe(
        source_datapipe)
    forever = True
    while forever:
        try:
            # Non-blocking call is Extremely slow here for python.mp, need to figureout good workaround
            request = req_queue.get(block=blocking_request_get)
        except:
            yield True
            continue

        if isinstance(request, datapipes.nonblocking.ResetIteratorRequest):
            source_datapipe.reset_iterator()
            res_queue.put(datapipes.nonblocking.ResetIteratorResponse())

        elif isinstance(request, datapipes.nonblocking.StopIteratorRequest):
            forever = False
            res_queue.put(datapipes.nonblocking.StopIteratorResponse())

        elif isinstance(request, datapipes.nonblocking.LenRequest):
            while forever:
                try:
                    value = source_datapipe.nonblocking_len()
                except datapipes.nonblocking.NotAvailable:
                    yield True
                    continue
                res_queue.put(datapipes.nonblocking.LenResponse(value), block=True)
                yield True  # Returns control
                break

        elif isinstance(request, datapipes.nonblocking.GetItemRequest):
            while forever:
                try:
                    value = source_datapipe.nonblocking_get(request.key)
                except datapipes.nonblocking.NotAvailable:
                    yield True
                    continue
                except StopIteration:
                    res_queue.put(StopIteration())
                    if full_stop:
                        forever = False
                    else:
                        yield True
                    break
                res_queue.put(datapipes.nonblocking.GetItemResponse(request.key, value), block=True)
                yield True  # Returns control
                break

        else:
            raise Exception('Unrecognized request', request)
