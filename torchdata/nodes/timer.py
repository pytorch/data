import collections
import time


class _Timer:
    def __init__(self):
        self.calls = 0
        self.dt = 0.0

    def __enter__(self):
        self.tstart = time.perf_counter()
        self.calls += 1
        return self

    def __exit__(self, type, value, traceback):
        self.dt += time.perf_counter() - self.tstart
        return False


def Timer():
    return collections.defaultdict(_Timer)
