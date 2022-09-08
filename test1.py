import multiprocessing as mp
import os

# from torchdata.dataloader2 import communication
import threading

import time

from torchdata.dataloader2 import (
    communication,
    DataLoader2,
    MultiProcessingReadingService,
    Prototype2MultiProcessingReadingService,
    PrototypeMultiProcessingReadingService,
)
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe


def inc(x):
    return x + 1


def is_odd(x):
    return bool(x % 2)


class PrefetchData:
    def __init__(self, source_datapipe, prefetch):
        self.run_prefetcher = True
        self.prefetch_buffer = []
        self.prefetch = prefetch
        self.source_datapipe = source_datapipe


class PrefetcherIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, prefetch=10):
        self.source_datapipe = source_datapipe
        self.prefetch = prefetch
        self.thread = None

    @staticmethod
    def thread_worker(prefetch_data):
        print(os.getpid(), "!!!!!!!! thread starting")
        # time.sleep(10)
        # print('now creating iterator')
        itr = iter(prefetch_data.source_datapipe)
        print(os.getpid(), "iterator done")
        stop_iteration = False
        while prefetch_data.run_prefetcher:
            if len(prefetch_data.prefetch_buffer) < prefetch_data.prefetch and not stop_iteration:
                try:
                    print(os.getpid(), "thread getting item")
                    # if prefetch_data.run_prefetcher:
                    item = next(itr)
                    print(os.getpid(), "thread getting item complete")
                    prefetch_data.prefetch_buffer.append(item)
                    print(os.getpid(), "item received and store in buffer of ", len(prefetch_data.prefetch_buffer))
                except (
                    RuntimeError,
                    StopIteration,
                ):  # TODO(VitalyFedyunin): Instead of general exception catch invalid iterator here
                    stop_iteration = True
                except communication.iter.InvalidStateResetRequired:
                    stop_iteration = True
                except communication.iter.TerminateRequired:
                    prefetch_data.run_prefetcher = False
            if stop_iteration and len(prefetch_data.prefetch_buffer) == 0:
                print(os.getpid(), "all items done, leaving thread myself")
                prefetch_data.run_prefetcher = False
            # print(os.getpid(),'thread wait with full buffer')
            time.sleep(0.00001)
        print(os.getpid(), "!!!!!!!!  exiting prefetch thread")

    def __iter__(self):
        self.reset()
        # self.run_prefetcher = True
        print(os.getpid(), ">>>>>>>> start thread")
        prefetch_data = PrefetchData(self.source_datapipe, self.prefetch)
        self.prefetch_data = prefetch_data
        self.thread = threading.Thread(target=PrefetcherIterDataPipe.thread_worker, args=(prefetch_data,), daemon=True)
        self.thread.start()
        i = 0
        while prefetch_data.run_prefetcher:
            if len(prefetch_data.prefetch_buffer) > 0:
                print(os.getpid(), "main loop returns item from buffer")
                yield prefetch_data.prefetch_buffer[0]
                prefetch_data.prefetch_buffer = prefetch_data.prefetch_buffer[1:]
            else:
                # print('waiting element {}-th time'.format(i))
                # i += 1
                time.sleep(0.00001)
        prefetch_data.run_prefetcher = False
        self.thread.join()
        self.thread = None

    def reset(self):
        print(os.getpid(), "resetting datapipe")
        if "terminate" in os.environ:
            raise Exception(os.getpid(), "who did it?")
        if self.thread is not None:
            self.prefetch_data.run_prefetcher = False
            self.thread.join()
        print(os.getpid(), "Reset complete")

    def reset_iterator(self):
        print(os.getpid(), "reset_iterator called on prefetcher")
        self.reset()


class RangeDebug:
    def __init__(self, x):
        self.x = x

    def __iter__(self):
        for i in range(self.x):
            print(os.getpid(), f">>>>>>>> returning {i}")
            yield i


def post_adapter_fn(dp):
    return PrefetcherIterDataPipe(dp, 10)


def main():
    items = 100000
    dp = IterableWrapper(RangeDebug(items)).filter(is_odd).sharding_filter()
    dp = PrefetcherIterDataPipe(dp, 5)
    dp = dp.map(slow_map)

    rs = Prototype2MultiProcessingReadingService(num_workers=2, post_adapter_fn=post_adapter_fn)
    dl = DataLoader2(dp, reading_service=rs)

    res = []

    start = time.time()

    for i, j in zip(iter(dl), range(4)):
        res.append(i)
    # time.sleep(3)
    print(os.getpid(), "-----------------------------------------------")

    # print(os.getpid(), 'creating iterator')
    # it = iter(dl)
    # print(os.getpid(), 'iterating')
    # item = next(it)
    # print(item)
    # os.environ['terminate'] = '1'
    for i in dl:
        # time.sleep(3)
        res.append(i)

    total = time.time() - start

    speed = items / total
    print(f"{speed} items per sec")

    # print(res)


def slow_map(x):
    # time.sleep(1)
    return x


def main_test():
    items = 1000
    dp = IterableWrapper(RangeDebug(items)).map(slow_map).filter(is_odd).sharding_filter()
    dp = PrefetcherIterDataPipe(dp, 20)
    dp = dp.map(slow_map)
    for i in iter(dp):
        print(i)


def main_mp_test():
    items = 1000
    datapipe = IterableWrapper(RangeDebug(items))

    ctx = mp.get_context("fork")
    num_workers = 2
    forked_dps = datapipe.fork(num_workers)

    sharded_forked_dps = []
    # Manually add sharding filters (per forked pipe), and apply sharding
    for pipe_id, pipe in enumerate(forked_dps):
        sharded_dp = pipe.sharding_filter()
        sharded_dp.apply_sharding(num_workers, pipe_id)
        sharded_forked_dps.append(sharded_dp)
    call_inside_process = None  # functools.partial(self.init_datapipe_process, 1, 0)
    process, pipes_and_queues = communication.eventloop.SpawnProcessForMultipleDataPipelines(
        ctx, sharded_forked_dps, call_inside_process
    )
    process.start()

    processes = []
    datapipes = []

    # Take care about termination of the separate process
    for _, req_queue, res_queue in pipes_and_queues:
        dp = communication.iter.QueueWrapper(
            communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
        )
        datapipes.append(dp)
        processes.append((process, req_queue, res_queue))

    # print(datapipes)

    dp0, dp1 = datapipes
    print(os.getpid(), "creating iterator 0")
    it0 = iter(dp0)
    print(os.getpid(), "creating iterator 1")
    it1 = iter(dp1)
    print(os.getpid(), "next(it0)")
    item0 = next(it0)
    print(item0)
    item1 = next(it1)
    print(item1)
    item0 = next(it0)
    item0 = next(it0)
    item1 = next(it1)
    print(os.getpid(), "resetting it1")
    it1 = iter(dp1)
    print(os.getpid(), "getting from it1")
    item1 = next(it1)
    print(os.getpid(), "getting from it0")
    try:
        item0 = next(it0)
    except communication.iter.InvalidStateResetRequired:
        print(os.getpid(), "invalid iterator confirmed")
    print(os.getpid(), "creating new iterator")
    it0 = iter(dp0)
    print(os.getpid(), "getting new item")
    item0 = next(it0)

    for process, req_queue, res_queue in processes:
        req_queue.put(communication.messages.TerminateRequest())

    process.join()


main()
