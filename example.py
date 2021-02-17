import json
import time
import multiprocessing
import types
import copy
import os
import ds

import datapipes
import dataloader

DELAY = 0.0001
# TOTAL_NUMBERS = 3

def stub_unpickler():
    return "STUB"

# Sends IterDataset to separate process with mp.Queues as connectors
def IterDatasetAsProcess(source_datapipe):
    pass
    

def PrintItems(ds):
    n = time.time()
    ds = datapipes.iter.EnsureNonBlockingNextDataPipe(ds)
    count = 0
    while True:
        try:
            value = ds.nonblocking_next()
            print(value)
            count += 1
        except StopIteration:
            print('time', time.time() - n, 'items', count)
            # print(dataloader.queue.LocalQueue.ops, dataloader.queue.LocalQueue.empty, dataloader.queue.LocalQueue.stored)
            break
        except datapipes.nonblocking.NotAvailable:
            # time.sleep(DELAY)
            dataloader.eventloop.EventLoop.iteration()
            
# ? WORK IN PROGRESS
def WrapDatasetToEvenHandlerKeepingObject(source_datapipe):
    source_datapipe = datapipes.iter.EnsureNonBlockingNextDataPipe(source_datapipe)
    def async_next(self):
        if self._as_iterator is None:
            self._as_iterator = iter(self)
        return next(self._as_iterator)
    setattr(source_datapipe, 'async_next', async_next)
    source_datapipe.async_next = types.MethodType(async_next, source_datapipe)
    pass

def example_1():
    ds0 = ds.NumbersDataset()
    q_ds0 = dataloader.eventloop.WrapDatasetToEventHandler(ds0, 'NumbersDataset')
    joined_ds = datapipes.iter.GreedyJoin(q_ds0)
    q_joined_ds = dataloader.eventloop.WrapDatasetToEventHandler(joined_ds, 'GreedyJoin')
    PrintItems(q_joined_ds)

# example_1()

def SpawnProcessForDataPipeGraph(dp):
    req_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=dataloader.eventloop.IterDataPipeToQueuesLoop, args=(dp, req_queue, res_queue))
    return process, req_queue, res_queue

def example_many_workers():
    num_workers = 6
    all_pipes = []
    all_workers = []
    for i in range(num_workers):
        numbers_dp = ds.NumbersDataset()
        shard_dp = datapipes.iter.SimpleSharding(numbers_dp)
        shard_dp.sharding_settings(num_workers, i)
        (process, req_queue, res_queue) = SpawnProcessForDataPipeGraph(shard_dp)
        process.start()
        all_workers.append(process)
        local_datapipe = datapipes.iter.QueueWrapper(req_queue, res_queue)
        all_pipes.append(local_datapipe)
    
    joined_dp = datapipes.iter.GreedyJoin(*all_pipes)
    # joined_dp = all_pipes[0]
    PrintItems(joined_dp)

    for worker in all_workers:
        worker.join()

def is_even(data):
    return data % 2 == 0

def is_odd(data):
    return data % 2 == 1

def slow_mult(x):
    return x * 100

def mult111(x):
    return x*111

def test_reset_iterator():
    numbers_dp = ds.NumbersDataset()
    wrapped_numbers_dp = dataloader.eventloop.WrapDatasetToEventHandler(numbers_dp, 'NumbersDataset')
    for item in wrapped_numbers_dp:
        print(item)
    print('----')
    for item in wrapped_numbers_dp:
        print(item)
    
test_reset_iterator()

def example_router():
    numbers_dp = ds.NumbersDataset()
    # numbers_dp = dataloader.eventloop.WrapDatasetToEventHandler(numbers_dp, 'NumbersDS')
    (even_dp, odd_dp) = datapipes.iter.Router(numbers_dp, [is_even, is_odd])
    # even_dp = dataloader.eventloop.WrapDatasetToEventHandler(even_dp, 'EvenDP')
    odd_dp = dataloader.eventloop.WrapDatasetToEventHandler(odd_dp, 'OddDP')
    updated_even_dp = datapipes.iter.Callable(even_dp, slow_mult)
    updated_even_dp = dataloader.eventloop.WrapDatasetToEventHandler(updated_even_dp, 'slow_mult')
    joined_dp = datapipes.iter.GreedyJoin(updated_even_dp, odd_dp)
    joined_dp = dataloader.eventloop.WrapDatasetToEventHandler(joined_dp, 'JoinedDP')
    even_more_dp = datapipes.iter.Callable(joined_dp, slow_mult)
    # even_more_dp = dataloader.eventloop.WrapDatasetToEventHandler(even_more_dp, 'MultAgain')
    for item in even_more_dp:
        print(item)
    print('----')
    for item in even_more_dp:
        print(item)

# example_router()

def example_2():
    ds0 = ds.NumbersDataset()
    p_ds = datapipes.iter.Prefetcher(ds0)
    
    pp_ds = dataloader.eventloop.WrapDatasetToEventHandler(p_ds, 'PrefetcherAsyncIterDataset')
    
    (ds1, ds2, ds3) = datapipes.iter.Multiply(pp_ds, 3)

    q_ds1 = dataloader.eventloop.WrapDatasetToEventHandler(ds1, 'MultiplyIterDatasetList_1')
    q_ds2 = dataloader.eventloop.WrapDatasetToEventHandler(ds2, 'MultiplyIterDatasetList_2')
    q_ds3 = dataloader.eventloop.WrapDatasetToEventHandler(ds3, 'MultiplyIterDatasetList_3')

    ds2_v2 = datapipes.iter.Callable(q_ds2, slow_mult)
    q_ds2_v2 = dataloader.eventloop.WrapDatasetToEventHandler(ds2_v2, 'datapipes.iter.Callable')

    # if True:
    #     req_queue = multiprocessing.Queue()
    #     res_queue = multiprocessing.Queue()
    #     pr2 = multiprocessing.Process(target=IterDatasetToQueuesLoop, args=(ds2_v2, req_queue, res_queue))
    #     pr2.start()
    #     q_ds2_v2 = QIterDataset(req_queue, res_queue)
    # else:
    #     q_ds2_v2 = dataloader.eventloop.WrapDatasetToEventHandler(ds2_v2, 'datapipes.iter.Callable')


    ds3_v2 = datapipes.iter.Callable(q_ds3, mult111)
    q_ds3_v2 = dataloader.eventloop.WrapDatasetToEventHandler(ds3_v2, 'datapipes.iter.Callable2')

    joined_ds = datapipes.iter.GreedyJoin(q_ds1, q_ds2_v2, q_ds3_v2)
    q_joined_ds = dataloader.eventloop.WrapDatasetToEventHandler(joined_ds, 'GreedyJoinIterDataset')

    # PrintItems(q_joined_ds)
    return q_joined_ds

# example_2()

def example_3():
    import pickle
    import copyreg
    import io



    # print(scanned)
    # print(p.dispatch_table)
    # p.dispatch_table[QIterDataset] = stub_pickler
    # p.dispatch_table[GreedyJoinIterDataset] = stub_pickler
    
    # p.dispatch_table[type(req_queue)] = stub_pickler

    def list_connected_di(scan_obj):

        f = io.BytesIO()
        p = pickle.Pickler(f)
        p.dispatch_table = copyreg.dispatch_table.copy()
        # scanned = []

        def stub_pickler(obj):
            return stub_unpickler, ()

        captured_connections = []
        def reduce_hook(obj):
            if obj == scan_obj:
                raise NotImplementedError
            else:
                captured_connections.append(obj)
                return stub_unpickler, ()
                # print('reduce hook reporting', obj)

        dataloader.graph.reduce_ex_hook = reduce_hook
        p.dump(scan_obj)
        # print(joined_ds)
        ds.reduce_ex_hook = None
        return captured_connections


    def traverse(ds):
        def make_key(item):
            return "%s id:%s" % (item.__class__, id(item))
        items = list_connected_di(ds)
        key = make_key(ds)
        d = { key : {} }
        for item in items:
            d[key].update(traverse(item))
        return d

    # items = list_connected_di(ds1)

    joined_ds = example_2() 
    # print(list_connected_di(joined_ds))
    structure = traverse(joined_ds)
    text = json.dumps(structure, indent=4)
   
    print(text)
    PrintItems(joined_ds)
    # connections = { joined_ds : items }
    
    # for item in items:
    #     connections[item] = list_connected_di(item)

    # # print(connections[ds0])

    
    # exc = set()
    # ins = {}
    # for k, v in connections.items():
    #     count = 0
    #     for item in v:            
    #         if item not in exc:
    #             count += 1
    #     ins[k] = count
    
    # print(ins)
                

# example_3()