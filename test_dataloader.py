import unittest
import timeout_decorator

import torch.utils.data
from torch.utils.data import IterDataPipe, IterableDataset, Dataset
from torch.utils.data.datapipes.iter import Map, Filter
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


class MapNumbersDataset(Dataset):
    def __init__(self, size=TOTAL_NUMBERS):
        self.size = size

    def __getitem__(self, key):
        return key * 10

    def __len__(self):
        return self.size


# This is fake class until we implement `.map` for Map style DataPipes
class MapMapDataPipe(Dataset):
    def __init__(self, source_dp, fn):
        self.source_dp = source_dp
        self.fn = fn

    def __getitem__(self, key):
        return self.fn(self.source_dp[key])

    def __len__(self):
        return self.source_dp.__len__()


class SumMapDataPipe(Dataset):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __getitem__(self, key):
        return self.a[key] + self.b[key]

    def __len__(self):
        return min(len(self.a), len(self.b))


def is_even(data):
    return data % 2 == 0


def is_odd(data):
    return data % 2 == 1


def mult_100(x):
    return x * 100


class TestClass(unittest.TestCase):
    def test_mapdataset(self):
        numbers_dp = MapNumbersDataset(size=10)
        numbers_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            numbers_dp, 'NumbersDataset_1')
        numbers_dp = MapMapDataPipe(numbers_dp, lambda x: x + 1)

        numbers_dp_2 = MapNumbersDataset(size=10)
        numbers_dp_2 = dataloader.eventloop.WrapDatasetToEventHandler(
            numbers_dp_2, 'NumbersDataset_2')

        sum_dp = SumMapDataPipe(numbers_dp, numbers_dp_2)

        actual = [sum_dp[i] for i in range(len(sum_dp))]
        expected = [i * 20 + 1 for i in range(10)]

        self.assertEqual(actual, expected)

    def test_reset_iterator(self):
        numbers_dp = NumbersDataset(size=10)
        wrapped_numbers_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            numbers_dp, 'NumbersDataset')

        items = []
        items = items + list(wrapped_numbers_dp)
        items = items + list(wrapped_numbers_dp)

        self.assertEqual(
            sorted(items), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])

    def test_functional(self):
        numbers_dp = NumbersDataset(size=10).filter(
            filter_fn=lambda x: x % 2 == 1).map(fn=lambda x: x * 10)
        actual = [i for i in numbers_dp]
        self.assertEqual(actual, [10, 30, 50, 70, 90])

    @timeout_decorator.timeout(5)
    def test_router_datapipe(self):
        numbers_dp = NumbersDataset(size=10)
        (even_dp, odd_dp) = datapipes.iter.Router(
            numbers_dp, [is_even, is_odd])
        odd_dp = dataloader.eventloop.WrapDatasetToEventHandler(odd_dp, 'Odd', prefetch = True)
        updated_even_dp = Map(even_dp, fn=mult_100)
        updated_even_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            updated_even_dp, 'MultipliedEven', prefetch = True)
        joined_dp = updated_even_dp.join(odd_dp)
        joined_dp = dataloader.eventloop.WrapDatasetToEventHandler(
            joined_dp, 'JoinedDP', prefetch = True)
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
            process = ctx.Process(target=dataloader.eventloop.DataPipeToQueuesLoop, args=(
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
            local_datapipe = datapipes.iter.QueueWrapper(
                dataloader.queue.IterDataPipeQueueProtocol(req_queue, res_queue))
            all_pipes.append(local_datapipe)

            cleanup_fn_args.append((req_queue, res_queue, process, i))

        joined_dp = datapipes.iter.GreedyJoin(*all_pipes)

        items = list(joined_dp)
        items += list(joined_dp)  # Reiterate second time

        for args in cleanup_fn_args:
            clean_me(*args)

        expected = list(range(50)) + list(range(50))

        self.assertEqual(sorted(items), sorted(expected))

    @timeout_decorator.timeout(60)
    def test_multiple_multiprocessing_workers_map_dataset(self):

        num_workers = 6
        all_pipes = []
        cleanup_fn_args = []
        ctx = multiprocessing.get_context('fork')

        def clean_me(req_queue, res_queue, process, pid):
            req_queue.put(datapipes.nonblocking.StopIteratorRequest())
            _ = res_queue.get()
            process.join()

        for i in range(num_workers):
            numbers_dp = MapNumbersDataset(size=50)
            (process, req_queue, res_queue) = dataloader.eventloop.SpawnProcessForDataPipeline(ctx, numbers_dp)
            process.start()
            # TODO(VitalyFedyunin): This is prone to error, do IterProtocol and MapProtocol to join Queue couples
            local_datapipe = datapipes.map.QueueWrapper(dataloader.queue.MapDataPipeQueueProtocol(req_queue, res_queue))
            all_pipes.append(local_datapipe)
            cleanup_fn_args.append((req_queue, res_queue, process, i))

        total_dp = MapNumbersDataset(size=50)

        for dp in all_pipes:
            total_dp = SumMapDataPipe(total_dp, dp)

        items = [total_dp[i] for i in range(len(total_dp))]
        expected = [i * 70 for i in range(50)]

        for args in cleanup_fn_args:
            clean_me(*args)

        self.assertEqual(items, expected)

    def _test_graph(self):
        numbers_dp = NumbersDataset(size=50)
        mapped_dp = Map(numbers_dp, mult_100)
        graph = dataloader.graph.traverse(mapped_dp)
        expected = {mapped_dp: {numbers_dp: {}}}
        self.assertEqual(graph, expected)

    def test_determinism(self):
        num_dp1 = NumbersDataset(size=50)
        num_dp2 = NumbersDataset(size=50)
        self.assertEqual(torch.utils.data.decorator._determinism, False)
        # Determinism guaranteed
        with torch.utils.data.guaranteed_datapipes_determinism():
            self.assertEqual(torch.utils.data.decorator._determinism, True)
            # Error thrown at the construction time
            # Sequential API
            with self.assertRaises(TypeError):
                joined_dp = datapipes.iter.GreedyJoin(num_dp1, num_dp2)
            # Functional API
            with self.assertRaises(TypeError):
                joined_dp = num_dp1.join(num_dp2)
            # With deterministic_fn
            joined_dp = datapipes.iter.GreedyJoin(num_dp1)
            self.assertEqual(sorted(list(joined_dp)), list(range(50)))
            joined_dp = num_dp1.join()
            self.assertEqual(sorted(list(joined_dp)), list(range(50)))
        # Determinism not guaranteed
        self.assertEqual(torch.utils.data.decorator._determinism, False)
        joined_dp = num_dp1.join(num_dp2)
        import itertools
        exp = list(itertools.chain.from_iterable(itertools.repeat(i, 2) for i in range(50)))
        self.assertEqual(sorted(list(joined_dp)), exp)


if __name__ == '__main__':
    unittest.main()
