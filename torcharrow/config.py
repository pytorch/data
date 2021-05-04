import threading
from enum import Enum

# ------------------------------------------------------------------------------
# Congig values 



class Device(Enum):
    numpy = 'numpy'
    koski = 'koski'
    velox = 'velox'
    cudf = 'cudf'


class Trace(Enum):
    on = 1
    off = 0

# ------------------------------------------------------------------------------
# Config data: either global and constant or thread local and possibly mutable 
gconfig = {}

lconfig = threading.local()

# here is nested handling of traces and devices...
lconfig.device_stack = [Device.numpy]
lconfig.tracing = Trace.off


class run_on(object):
    def __init__(self, device):
        lconfig.device_stack.append(device)
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        del lconfig.device_stack[-1]
    @static_method
    def peek():
        return lconfig.device_stack.peek()

# example code
# with run_on(Device.koski):
# ... code...



      