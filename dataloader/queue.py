

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
            raise Exception('LocalQueue is empty')
        LocalQueue.stored -= 1
        return self.items.pop()
