import datapipes

class Protocol(object):
    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue

class ProtocolClient(Protocol):
    _req_sent = False

    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._req_sent = False

    def can_take_request(self):
        return not self._req_sent

    def request_sent(self):
        if self._req_sent:
            raise Exception('Protocol only supports one request in the Queue')
        self._req_sent = True

    def request_served(self):
        if self._req_sent:
            raise Exception(
                'Expected no peding requests, but something got served')
        self._req_sent = False

class MapDataPipeQueueProtocol(Protocol):
    pass


class IterDataPipeQueueProtocol(ProtocolClient):
    def request_reset(self):
        self.request_queue.put(datapipes.nonblocking.ResetIteratorRequest())
        self.request_sent()

    # def get_reset(self):
    #     try:
    #         value = self._res_q.get(block=False)
    #     except:
    #         raise Exception('Response not available')
    #     self.request_served()

            

class LocalQueue():
    ops = 0
    stored = 0
    uid = 0
    empty = 0
    allq = []

    @classmethod
    def report(cls):
        print('-')
        for q in cls.allq:
            print('queue', q.name, q.items)

    def __init__(self, name='unnamed'):
        self.items = []
        self.name = name
        self.uid = LocalQueue.uid
        LocalQueue.uid += 1
        LocalQueue.allq.append(self)

    def put(self, item, block=True):
        LocalQueue.ops += 1
        LocalQueue.stored += 1
        self.items.append(item)

    def get(self, block=True, timeout=0):
        # TODO(VitalyFedyunin): Add support of block and timeout arguments
        LocalQueue.ops += 1
        if not len(self.items):
            LocalQueue.empty += 1
            raise Exception('not available')
        LocalQueue.stored -= 1
        return self.items.pop()
