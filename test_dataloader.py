import unittest
import timeout_decorator

from torch.utils.data import IterDataPipe, IterableDataset
from torch.utils.data.datapipes.iter import Map
import torch.multiprocessing as multiprocessing

import datapipes
import dataloader

TOTAL_NUMBERS = 100


class NumbersDataset(IterableDataset):
    def __init__(self, size=TOTAL_NUMBERS):
        self.size = size

    def __iter__(self):
        for i in range(self.size):
            yield i


def is_even(data):
    return data % 2 == 0


def is_odd(data):
    return data % 2 == 1


def mult_100(x):
    return x * 100


class TestClass(unittest.TestCase):
    def test_reset_iterator(self):
        numbers_dp = NumbersDataset(size=10)
        wrapped_numbers_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            numbers_dp, 'NumbersDataset')

        items = []
        items = items + list(wrapped_numbers_dp)
        items = items + list(wrapped_numbers_dp)

        self.assertEqual(
            sorted(items), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])

    def test_router_datapipe(self):
        numbers_dp = NumbersDataset(size=10)
        (even_dp, odd_dp) = datapipes.iter.Router(
            numbers_dp, [is_even, is_odd])
        odd_dp = dataloader.eventloop.WrapDatasetToEventHandler(odd_dp, 'Odd')
        updated_even_dp = Map(even_dp, fn=mult_100)
        updated_even_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            updated_even_dp, 'MultipliedEven')
        joined_dp = datapipes.iter.GreedyJoin(updated_even_dp, odd_dp)
        joined_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            joined_dp, 'JoinedDP')
        items = list(joined_dp)
        self.assertEqual(sorted(items), [0, 1, 3, 5, 7, 9, 200, 400, 600, 800])
        
    def test_multiply_datapipe(self):
        numbers_dp = NumbersDataset(size=10)
        (one, two, three) = datapipes.iter.Multiply(
            numbers_dp, 3)
        joined_dp = datapipes.iter.GreedyJoin(one, two, three)
        joined_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            joined_dp, 'JoinedDP')
        items = list(joined_dp) + list(joined_dp)
        expected = list(range(10)) * 3 * 2
        self.assertEqual(sorted(items), sorted(expected))

    def test_router_datapipe_wrong_priority_fns(self):
        numbers_dp = NumbersDataset(size=10)
        (even_dp, odd_dp) = datapipes.iter.Router(
            numbers_dp, [is_even, is_even])
        odd_dp = dataloader.eventloop.WrapDatasetToEventHandler(odd_dp, 'Odd')
        updated_even_dp = Map(even_dp, fn=mult_100)
        updated_even_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            updated_even_dp, 'MultipliedEven')
        joined_dp = datapipes.iter.GreedyJoin(updated_even_dp, odd_dp)
        joined_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            joined_dp, 'JoinedDP')

        with self.assertRaises(Exception):
            _ = list(joined_dp)

    @timeout_decorator.timeout(5)
    def _test_router_datapipe_iterate_multiple_times(self):
        numbers_dp = NumbersDataset(size=10)
        (even_dp, odd_dp) = datapipes.iter.Router(
            numbers_dp, [is_even, is_odd])
        odd_dp = dataloader.eventloop.WrapDatasetToEventHandler(odd_dp, 'Odd')
        updated_even_dp = Map(even_dp, fn=mult_100)
        updated_even_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            updated_even_dp, 'MultipliedEven')
        joined_dp = datapipes.iter.GreedyJoin(updated_even_dp, odd_dp)
        joined_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            joined_dp, 'JoinedDP')
        items = list(joined_dp)
        items += list(joined_dp)
        expected = [0, 1, 3, 5, 7, 9, 200, 400, 600, 800] * 2
        self.assertEqual(sorted(items), sorted(expected))

    @timeout_decorator.timeout(60)
    def test_multiple_multiprocessing_workers(self):

        def SpawnProcessForDataPipeGraph(ctx, dp):
            req_queue = ctx.Queue()
            res_queue = ctx.Queue()
            process = ctx.Process(target=dataloader.eventloop.IterDataPipeToQueuesLoop, args=(
                dp, req_queue, res_queue))
            return process, req_queue, res_queue

        num_workers = 6
        all_pipes = []
        cleanup_fn_args = []
        ctx = multiprocessing.get_context('fork')

        def clean_me(req_queue, res_queue, process, pid):
            req_queue.put(datapipes.nonblocking.StopIteratorRequest())
            _ = res_queue.get()
            process.join()

        for i in range(num_workers):
            numbers_dp = NumbersDataset(size=50)
            shard_dp = datapipes.iter.SimpleSharding(numbers_dp)
            shard_dp.sharding_settings(num_workers, i)
            (process, req_queue, res_queue) = SpawnProcessForDataPipeGraph(ctx, shard_dp)
            process.start()
            local_datapipe = datapipes.iter.QueueWrapper(req_queue, res_queue)
            all_pipes.append(local_datapipe)

            cleanup_fn_args.append((req_queue, res_queue, process, i))

        joined_dp = datapipes.iter.GreedyJoin(*all_pipes)

        items = list(joined_dp)
        items += list(joined_dp)  # Reiterate second time

        for args in cleanup_fn_args:
            clean_me(*args)

        expected = list(range(50)) + list(range(50))

        self.assertEqual(sorted(items), sorted(expected))


if __name__ == '__main__':
    unittest.main()
