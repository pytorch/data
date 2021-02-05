class LocalQueue():
    ops = 0
    stored = 0
    uid = 0
    empty = 0 

    def __init__(self, name = 'unnamed'):
        self.items = []
        self.name = name
        self.uid = LocalQueue.uid
        LocalQueue.uid += 1
    
    def put(self, item, block = True):
        LocalQueue.ops += 1
        LocalQueue.stored += 1
        self.items.append(item)

    def get(self, block = True, timeout = 0):
        LocalQueue.ops += 1
        if not len(self.items):
            LocalQueue.empty += 1
            raise Exception('not available')
        LocalQueue.stored -= 1
        return self.items.pop()
        