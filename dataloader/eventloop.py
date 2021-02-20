import copy
import threading

import torch
import time

import datapipes
import dataloader.queue


class EventLoop:
    '''
    Threading and multi-processing will require own versions of EventLoop,
    this POC version doesn't support it.
    '''
    handlers = []
    loop_generators = None
    uid = 0
    stack = []
    depth = 0
    thread_local = None
    
    @classmethod
    def init(cls):
        cls.thread_local = threading.local()
        cls.thread_local.enabled = True
        cls.handlers = []
        cls.loop_generators = None
        cls.uid = 0
        cls.stack = []
        cls.depth = 0
            
    @classmethod 
    def is_enabled(cls):
        return cls.thread_local.enabled
    
    @classmethod
    def disable(cls):
        cls.thread_local.enabled = False
        
    @classmethod
    def iteration(cls):
        if not cls.is_enabled() or not len(cls.handlers):
            return
        if cls.loop_generators is None:
            cls.loop_generators = [iter(cls._loop_iterator())]
        if len(cls.loop_generators) < cls.depth+1:
            cls.loop_generators.append(iter(cls._loop_iterator()))
        try:
            cls.depth += 1
            next(cls.loop_generators[cls.depth - 1])
            cls.depth -= 1
        except Exception as e:
            print(e)
            raise

    @classmethod
    def _loop_iterator(cls):
        while True:
            for handle, _handle_name, uid in cls.handlers:
                try:
                    if uid not in cls.stack:
                        stack_len = len(cls.stack)
                        cls.stack.append(uid)
                        value = next(handle)
                        stack_copy = copy.deepcopy(cls.stack)
                        cls.stack.clear()
                        if stack_len:
                            for i in range(stack_len):
                                cls.stack.append(stack_copy[i])
                        yield value
                    else:
                        pass
                except Exception as e:
                    print(e)
                    raise

    @classmethod
    def add_handler(cls, handler, handle_name='unnamed'):
        cls.uid += 1
        cls.handlers.append((handler, handle_name, cls.uid))

# Turns IterDataPipe into two mp.Queues, terminates when getting StopIteration
def IterDataPipeToQueuesLoop(source_datapipe, req_queue, res_queue):
    torch.set_num_threads(1)
    # Stop EventLoop for MultiProcessing and Threading case
    EventLoop.disable()
    for _ in IterDataPipeBehindQueues(source_datapipe, req_queue, res_queue, full_stop=False, blocking_request_get=True):
        pass

# Indefinitely iterates over req_queue and passing values from source_datapipe to res_queue
# If raise_stop is true, raises exception when StopIteration received from the source_datapipe
def IterDataPipeBehindQueues(source_datapipe, req_queue, res_queue, full_stop=False, nonblocking_next_function_name='nonblocking_next', blocking_request_get=False):
    source_datapipe = datapipes.iter.EnsureNonBlockingNextDataPipe(
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
            continue

        if isinstance(request, datapipes.nonblocking.StopIteratorRequest):
            forever = False
            res_queue.put(datapipes.nonblocking.StopIteratorResponse())
            continue

        while forever:
            try:
                function = getattr(
                    source_datapipe, nonblocking_next_function_name)
                value = function()
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
            res_queue.put(value, block=True)
            yield True  # Returns control
            break

# Puts datapipe behind two (request, response) queues, adds Iterator to the EventLoop to process messages
def WrapDatasetToEventHandler(source_datapipe, dp_name='unnamed dataset'):
    source_datapipe = datapipes.iter.EnsureNonBlockingNextDataPipe(
        source_datapipe)
    # request_queue = multiprocessing.Queue()
    # response_queue = multiprocessing.Queue()
    request_queue = dataloader.queue.LocalQueue(name=dp_name + ' request')
    response_queue = dataloader.queue.LocalQueue(name=dp_name + ' response')

    handler = iter(IterDataPipeBehindQueues(source_datapipe, request_queue,
                                            response_queue, nonblocking_next_function_name='nonblocking_next'))
    EventLoop.add_handler(handler, dp_name)
    datapipe = datapipes.iter.QueueWrapper(request_queue, response_queue)
    datapipe._wrapped_source_datapipe = source_datapipe
    return datapipe


def SpawnProcessForDataPipeline(multiprocessing_ctx, datapipe):
    req_queue = multiprocessing_ctx.Queue()
    res_queue = multiprocessing_ctx.Queue()
    process = multiprocessing_ctx.Process(
        target=IterDataPipeToQueuesLoop, args=(datapipe, req_queue, res_queue))
    return process, req_queue, res_queue


def SpawnThreadForDataPipeline(datapipe):
    req_queue = dataloader.queue.ThreadingQueue()
    res_queue = dataloader.queue.ThreadingQueue()
    
    process = threading.Thread(target=IterDataPipeToQueuesLoop, args=(
        datapipe, req_queue, res_queue))
    return process, req_queue, res_queue

EventLoop.init()
datapipes.iter.NonBlocking.register_not_available_hook(
    lambda: EventLoop.iteration())
