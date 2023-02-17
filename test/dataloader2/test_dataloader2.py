# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import pickle
import queue
import random
import socket
import unittest

from unittest import TestCase

import numpy as np

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import instantiate_parametrized_tests, IS_WINDOWS, parametrize

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from torchdata.dataloader2 import (
    communication,
    DataLoader2,
    DistributedReadingService,
    MultiProcessingReadingService,
    ReadingServiceInterface,
    SequentialReadingService,
)
from torchdata.dataloader2.dataloader2 import READING_SERVICE_STATE_KEY_NAME, SERIALIZED_DATAPIPE_KEY_NAME

from torchdata.dataloader2.graph import DataPipe, list_dps, replace_dp, set_datapipes_seed, traverse_dps
from torchdata.dataloader2.random import SeedGenerator
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe, ShardingRoundRobinDispatcher
from torchdata.datapipes.map import SequenceWrapper

try:
    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

skipIfNoDill = unittest.skipIf(not HAS_DILL, "no dill")

if dist.is_available():
    HAS_DIST = True
else:
    HAS_DIST = False

skipIfNoDistributed = unittest.skipIf(not HAS_DIST, "no torch.distributed")

TEST_WITH_TSAN = os.getenv("PYTORCH_TEST_WITH_TSAN", "0") == "1"

mp_ctx_parametrize = parametrize("ctx", mp.get_all_start_methods())

EXCEPTION_ITERATION_NUM = 7


class _ReadingServiceWrapper:
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        self.it = iter(self.dp)
        return self

    def __next__(self):
        return next(self.it)

    @staticmethod
    def return_one():
        return 1


class TestReadingService(ReadingServiceInterface):
    def initialize(self, dp: DataPipe) -> DataPipe:
        return _ReadingServiceWrapper(dp)  # type: ignore[return-value]


class DataLoader2Test(TestCase):
    def test_dataloader2(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe)

        expected_batch = 0
        for batch in iter(data_loader):
            self.assertEqual(batch, expected_batch)
            expected_batch += 1

    def test_dataloader2_shutdown(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe)
        data_loader.shutdown()

    def test_dataloader2_state_dict(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe)

        state = data_loader.state_dict()
        self.assertIsNotNone(state)
        self.assertIsNotNone(state[SERIALIZED_DATAPIPE_KEY_NAME])
        self.assertIsNone(state[READING_SERVICE_STATE_KEY_NAME])
        data_loader.shutdown()

    def test_dataloader2_reading_service(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        reading_service = TestReadingService()
        data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe, reading_service=reading_service)

        expected_batch = 0
        for batch in iter(data_loader):
            self.assertEqual(batch, expected_batch)
            expected_batch += 1

    def test_dataloader2_load_state_dict(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        reading_service = TestReadingService()
        data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe, reading_service=reading_service)

        batch = next(iter(data_loader))
        self.assertEqual(batch, 0)

        state = data_loader.state_dict()
        self.assertIsNotNone(state)
        self.assertIsNotNone(state[SERIALIZED_DATAPIPE_KEY_NAME])
        self.assertIsNone(state[READING_SERVICE_STATE_KEY_NAME])
        data_loader.shutdown()

        restored_data_loader: DataLoader2 = DataLoader2(datapipe=None, reading_service=reading_service)
        restored_data_loader.load_state_dict(state)

        restored_data_loader_datapipe = restored_data_loader.datapipe
        deserialized_datapipe = pickle.loads(state[SERIALIZED_DATAPIPE_KEY_NAME])
        for batch_1, batch_2 in zip(restored_data_loader_datapipe, deserialized_datapipe):
            self.assertEqual(batch_1, batch_2)

        self.assertEqual(
            restored_data_loader.reading_service_state,
            state[READING_SERVICE_STATE_KEY_NAME],
        )

        restored_data_loader.shutdown()

    def test_dataloader2_iterates_correctly(self) -> None:
        test_data_pipe = IterableWrapper(range(10)).sharding_filter()
        reading_services = [
            None,
            TestReadingService(),
            MultiProcessingReadingService(num_workers=4),
            MultiProcessingReadingService(num_workers=4, worker_prefetch_cnt=0),
        ]
        for reading_service in reading_services:
            data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe, reading_service=reading_service)
            self.assertEqual(list(range(10)), list(data_loader))
            self.assertEqual(list(range(10)), list(data_loader))
            self.assertEqual(list(range(10)), list(data_loader))
            actual = []
            for i in data_loader:
                actual.append(i)
            self.assertEqual(list(range(10)), actual)
            actual = []
            for i in data_loader:
                actual.append(i)
            self.assertEqual(list(range(10)), actual)

    def test_dataloader2_reset(self) -> None:
        test_data_pipe = IterableWrapper(range(10))
        reading_services = [None, TestReadingService(), MultiProcessingReadingService(num_workers=1)]

        for reading_service in reading_services:
            data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe, reading_service=reading_service)

            # Functional Test: Ensure multiple sequential reads of DL2 is possible
            self.assertEqual(list(range(10)), list(data_loader))
            self.assertEqual(list(range(10)), list(data_loader))
            self.assertEqual(list(range(10)), list(data_loader))

            # Functional Test: Ensure that the creation of a new iterator invalidates the old one
            it1 = iter(data_loader)
            self.assertEqual(0, next(it1))
            self.assertEqual(1, next(it1))
            it2 = iter(data_loader)
            self.assertEqual(0, next(it2))
            self.assertEqual(1, next(it2))
            with self.assertRaisesRegex(RuntimeError, "iterator has been invalidated"):
                next(it1)
            self.assertEqual(list(range(2, 10)), list(it2))

    def test_dataloader2_delegate_attribute(self) -> None:
        test_data_pipe = IterableWrapper(range(10))
        data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe, reading_service=TestReadingService())

        # Functional Test: Ensure multiple sequential reads of DL2 is possible
        self.assertEqual(list(range(10)), list(data_loader))
        self.assertEqual(list(range(10)), list(data_loader))

        # Functional Test: Ensure that attribute/method of `dataloader._datapipe_iter` can be used
        it = iter(data_loader)
        self.assertEqual(1, it.return_one())  # type: ignore[attr-defined]


class DataLoader2ConsistencyTest(TestCase):
    r"""
    These tests ensure that the behaviors of `DataLoader2` are consistent across `ReadingServices` and potentially
    with `DataLoaderV1`.
    """

    @staticmethod
    def _get_no_reading_service():
        return None

    @staticmethod
    def _get_mp_reading_service():
        return MultiProcessingReadingService(num_workers=2)

    @staticmethod
    def _get_mp_reading_service_zero_workers():
        return MultiProcessingReadingService(num_workers=0)

    def _collect_data(self, datapipe, reading_service_gen):
        dl: DataLoader2 = DataLoader2(datapipe, reading_service=reading_service_gen())
        result = []
        # Testing how RS handles partial reading and reiterations
        for row, _ in zip(dl, range(10)):
            result.append(row)
        for row in dl:
            result.append(row)
        dl.shutdown()
        return result

    @staticmethod
    def _no_op(x):
        return x

    def test_dataloader2_batch_collate(self) -> None:
        dp: IterDataPipe = IterableWrapper(range(100)).batch(2).sharding_filter().collate(self._no_op)  # type: ignore[assignment]
        expected = self._collect_data(dp, reading_service_gen=self._get_no_reading_service)

        reading_service_generators = (
            self._get_mp_reading_service,
            self._get_mp_reading_service_zero_workers,
        )
        for reading_service_gen in reading_service_generators:
            actual = self._collect_data(dp, reading_service_gen=reading_service_gen)
            # TODO(588): This comparison only indicates that somethings is broken and not helping with debug
            self.assertEqual(expected, actual, reading_service_gen)

    def test_dataloader2_shuffle(self) -> None:
        # TODO(589): Add shuffle test
        pass


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestDataLoader2EventLoop(TestCase):
    # TODO: This needs fixing, see issue 624
    # @skipIfNoDill
    # def test_basic_threading(self):
    #     def clean_me(process, req_queue, res_queue):
    #         req_queue.put(communication.messages.TerminateRequest())
    #         _ = res_queue.get()
    #         process.join()
    #
    #     it = list(range(100))
    #     numbers_dp = IterableWrapper(it)
    #     (process, req_queue, res_queue, _thread_local_datapipe) = communication.eventloop.CreateThreadForDataPipeline(numbers_dp)
    #
    #     process.start()
    #     local_datapipe = communication.iter.QueueWrapper(
    #         communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue))
    #
    #     actual = list(local_datapipe)
    #     clean_me(process, req_queue, res_queue)
    #
    #     self.assertEqual(list(range(100)), actual)

    @skipIfNoDill
    def test_basic_mapdatapipe_threading(self):
        def clean_me(process, req_queue, res_queue):
            req_queue.put(communication.messages.TerminateRequest())
            _ = res_queue.get()
            process.join()

        input_len = 100
        it = list(range(input_len))
        numbers_dp = SequenceWrapper(it)
        (process, req_queue, res_queue, _thread_local_datapipe) = communication.eventloop.CreateThreadForDataPipeline(
            numbers_dp
        )

        process.start()

        # Functional Test: Ensure that you can retrieve every element from the Queue and DataPipe
        local_datapipe = communication.map.QueueWrapperForMap(
            communication.protocol.MapDataPipeQueueProtocolClient(req_queue, res_queue)
        )
        actual = list(local_datapipe)
        self.assertEqual([(x, x) for x in range(100)], actual)

        # Functional Test: raise Error when input
        local_datapipe = communication.map.QueueWrapperForMap(
            communication.protocol.MapDataPipeQueueProtocolClient(req_queue, res_queue)
        )
        with self.assertRaisesRegex(IndexError, "out of bound"):
            local_datapipe[1000]

        # __len__ Test: Ensure that the correct length is returned
        local_datapipe = communication.map.QueueWrapperForMap(
            communication.protocol.MapDataPipeQueueProtocolClient(req_queue, res_queue)
        )
        self.assertEqual(input_len, len(local_datapipe))

        clean_me(process, req_queue, res_queue)


def _x_mult_2(d):
    return d * 2


class NonReplicableDataPipe(IterDataPipe):
    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        yield from self.datapipe

    def is_replicable(self):
        return False


class _CustomException(Exception):
    pass


class MakeMistakeDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, exc_iteration=EXCEPTION_ITERATION_NUM):
        self.source_datapipe = source_datapipe
        self.exc_iteration = exc_iteration

    def __iter__(self):
        for i, x in enumerate(self.source_datapipe):
            if i == self.exc_iteration:
                raise _CustomException("oops")
            yield x


class MultiProcessingReadingServiceTest(TestCase):
    @staticmethod
    def _worker_init_fn(datapipe, worker_info):
        datapipe = datapipe.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(
            datapipe, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
        )
        return datapipe

    @staticmethod
    def _worker_reset_fn(datapipe, worker_info, worker_seed_generator: SeedGenerator):
        graph = traverse_dps(datapipe)
        dps = list_dps(graph)
        worker_seed_generator.seed(123)
        set_datapipes_seed(dps, seed_generator=worker_seed_generator, distributed_shared=True)
        return datapipe

    @mp_ctx_parametrize
    def test_worker_fns(self, ctx):
        dp: IterDataPipe = IterableWrapper(range(100)).batch(2).shuffle()

        rs = MultiProcessingReadingService(
            num_workers=2,
            multiprocessing_context=ctx,
            worker_init_fn=self._worker_init_fn,
            worker_reset_fn=self._worker_reset_fn,
        )
        dl = DataLoader2(dp, reading_service=rs)

        # Test worker_reset_fn to set the same random seed across epoches
        res1 = list(dl)
        res2 = list(dl)
        self.assertEqual(res1, res2)

    @mp_ctx_parametrize
    def test_single_branch_non_replicable(self, ctx):
        r"""
        For single branch pipeline with a non-replicable DataPipe, all ``sharding_filters``
        in the pipeline become non-replicable.
        """

        def _make_dp():
            single_br_dp = IterableWrapper(list(range(10))).shuffle()
            map_dp = single_br_dp.map(_x_mult_2)
            end_dp = map_dp.map(_x_mult_2).shuffle()
            return single_br_dp, map_dp, end_dp

        def _assert_deterministic_dl_res(dl, exp):
            torch.manual_seed(123)
            res = list(dl)
            self.assertEqual(sorted(res), exp)
            # Second epoch
            torch.manual_seed(123)
            self.assertEqual(list(dl), res)
            # Different seed
            torch.manual_seed(321)
            self.assertNotEqual(list(dl), res)
            # Properly shutdown
            dl.shutdown()

        # By-default, all replicable
        single_br_dp, _, end_dp = _make_dp()
        graph = traverse_dps(end_dp)
        sf_dp = single_br_dp.sharding_filter()
        replace_dp(graph, single_br_dp, sf_dp)
        dl = DataLoader2(
            end_dp, reading_service=MultiProcessingReadingService(num_workers=2, multiprocessing_context=ctx)
        )
        # Determinism and dynamic sharding
        #  _assert_deterministic_dl_res(dl, [i * 4 for i in range(10)])

        # Non-replicable before sharding_filter
        # shuffle in dispatch process
        single_br_dp, map_dp, end_dp = _make_dp()
        graph = traverse_dps(end_dp)
        round_robin_dispatcher = ShardingRoundRobinDispatcher(single_br_dp, SHARDING_PRIORITIES.MULTIPROCESSING)
        replace_dp(graph, single_br_dp, round_robin_dispatcher)
        sf_dp = map_dp.sharding_filter()
        replace_dp(graph, map_dp, sf_dp)
        dl = DataLoader2(
            end_dp, reading_service=MultiProcessingReadingService(num_workers=2, multiprocessing_context=ctx)
        )
        # Determinism for non-replicable pipeline
        _assert_deterministic_dl_res(dl, [i * 4 for i in range(10)])

        # Non-replicable after sharding_filter
        # shuffle in dispatch process
        single_br_dp, map_dp, end_dp = _make_dp()
        graph = traverse_dps(end_dp)
        sf_dp = single_br_dp.sharding_filter()
        replace_dp(graph, single_br_dp, sf_dp)
        round_robin_dispatcher = ShardingRoundRobinDispatcher(map_dp, SHARDING_PRIORITIES.MULTIPROCESSING)
        replace_dp(graph, map_dp, round_robin_dispatcher)
        dl = DataLoader2(
            end_dp, reading_service=MultiProcessingReadingService(num_workers=2, multiprocessing_context=ctx)
        )
        # Determinism for non-replicable pipeline
        _assert_deterministic_dl_res(dl, [i * 4 for i in range(10)])

    @mp_ctx_parametrize
    def test_multi_branch_non_replicable(self, ctx) -> None:
        r"""
        For multi-branch pipeline with a non-replicable DataPipe on one branch,
        all ``sharding_filter`` on the other branches should remain replicable.
        """

        def _make_dp():
            branch1_dp = IterableWrapper(list(range(10))).shuffle()
            branch2_dp = IterableWrapper(list(range(10))).shuffle()
            map_dp = branch1_dp.map(_x_mult_2)
            end_dp = map_dp.zip(branch2_dp)
            return branch1_dp, map_dp, branch2_dp, end_dp

        def _assert_deterministic_dl_res(dl, exp1, exp2):
            torch.manual_seed(123)
            res = list(dl)
            res1, res2 = list(zip(*res))
            self.assertEqual(sorted(res1), exp1)
            self.assertEqual(sorted(res2), exp2)
            # Second epoch
            torch.manual_seed(123)
            self.assertEqual(list(dl), res)
            # Different seed
            torch.manual_seed(321)
            self.assertNotEqual(list(dl), res)
            # Properly shutdown
            dl.shutdown()

        # By-default, all replicable
        branch1_dp, _, branch2_dp, end_dp = _make_dp()
        graph = traverse_dps(end_dp)
        sf1_dp = branch1_dp.sharding_filter()
        sf2_dp = branch2_dp.sharding_filter()
        replace_dp(graph, branch1_dp, sf1_dp)
        replace_dp(graph, branch2_dp, sf2_dp)
        dl = DataLoader2(
            end_dp, reading_service=MultiProcessingReadingService(num_workers=2, multiprocessing_context=ctx)
        )
        # Determinism and dynamic sharding
        _assert_deterministic_dl_res(dl, [i * 2 for i in range(10)], list(range(10)))

        # Non-replicable on one branch
        # shuffle in dispatch process
        branch1_dp, _, branch2_dp, end_dp = _make_dp()
        graph = traverse_dps(end_dp)
        non_replicable_dp = ShardingRoundRobinDispatcher(branch1_dp, SHARDING_PRIORITIES.MULTIPROCESSING)
        replace_dp(graph, branch1_dp, non_replicable_dp)
        # The other branch should has a sharding_filter to make data even
        sf_dp = branch2_dp.sharding_filter()
        replace_dp(graph, branch2_dp, sf_dp)
        dl = DataLoader2(
            end_dp, reading_service=MultiProcessingReadingService(num_workers=2, multiprocessing_context=ctx)
        )
        # Determinism for non-replicable pipeline
        _assert_deterministic_dl_res(dl, [i * 2 for i in range(10)], list(range(10)))

        # Non-replicable on both branches
        # shuffle in dispatch process
        branch1_dp, _, branch2_dp, end_dp = _make_dp()
        graph = traverse_dps(end_dp)
        non_replicable_dp1 = ShardingRoundRobinDispatcher(branch1_dp, SHARDING_PRIORITIES.MULTIPROCESSING)
        replace_dp(graph, branch1_dp, non_replicable_dp1)
        non_replicable_dp2 = ShardingRoundRobinDispatcher(branch2_dp, SHARDING_PRIORITIES.MULTIPROCESSING)
        replace_dp(graph, branch2_dp, non_replicable_dp2)
        dl = DataLoader2(
            end_dp, reading_service=MultiProcessingReadingService(num_workers=2, multiprocessing_context=ctx)
        )
        # Determinism for non-replicable pipeline
        _assert_deterministic_dl_res(dl, [i * 2 for i in range(10)], list(range(10)))

    @mp_ctx_parametrize
    def test_multi_worker_determinism(self, ctx):
        dp: IterDataPipe = IterableWrapper(range(100))
        dp = dp.shuffle().sharding_filter()
        dp = dp.batch(2)

        rs = MultiProcessingReadingService(
            num_workers=2,
            multiprocessing_context=ctx,
        )
        dl = DataLoader2(dp, reading_service=rs)

        torch.manual_seed(123)
        res = list(dl) + list(dl)

        torch.manual_seed(123)
        self.assertEqual(res, list(dl) + list(dl))

        torch.manual_seed(321)
        self.assertNotEqual(res, list(dl) + list(dl))

        # Using seed API for DataLoader2
        dl.seed(123)
        res = list(dl) + list(dl)

        dl.seed(123)
        self.assertEqual(res, list(dl) + list(dl))

        dl.seed(321)
        self.assertNotEqual(res, list(dl) + list(dl))

    @mp_ctx_parametrize
    def test_dispatching_worker_determinism(self, ctx):
        dp: IterDataPipe = IterableWrapper(range(101))
        dp = dp.shuffle().sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
        dp = dp.batch(2)

        rs = MultiProcessingReadingService(
            num_workers=2,
            multiprocessing_context=ctx,
        )
        dl = DataLoader2(dp, reading_service=rs)

        torch.manual_seed(123)
        res = list(dl) + list(dl)

        torch.manual_seed(123)
        self.assertEqual(res, list(dl) + list(dl))

        torch.manual_seed(321)
        self.assertNotEqual(res, list(dl) + list(dl))

        # Using seed API for DataLoader2
        dl.seed(123)
        res = list(dl) + list(dl)

        dl.seed(123)
        self.assertEqual(res, list(dl) + list(dl))

        dl.seed(321)
        self.assertNotEqual(res, list(dl) + list(dl))

    @mp_ctx_parametrize
    def test_non_replicable_datapipe(self, ctx) -> None:
        r"""
        For the pipeline with non-replicable DataPipe, make sure
        the DataPipe remains in the main process.
        """
        dp: IterDataPipe = IterableWrapper(range(100))
        dp = dp.shuffle().sharding_filter()
        dp = dp.batch(2)
        non_rep_dp = NonReplicableDataPipe(dp)

        rs = MultiProcessingReadingService(
            num_workers=2,
            multiprocessing_context=ctx,
        )
        dl = DataLoader2(non_rep_dp, reading_service=rs)

        torch.manual_seed(123)
        it = iter(dl)
        # Validate NonReplicableDataPipe still in the main process
        non_rep_dp = dl.reading_service._end_datapipe
        self.assertEqual(type(non_rep_dp), NonReplicableDataPipe)

        res = list(it) + list(dl)

        torch.manual_seed(123)
        self.assertEqual(res, list(dl) + list(dl))

        torch.manual_seed(321)
        self.assertNotEqual(res, list(dl) + list(dl))

    @parametrize("num_workers", [1, 3])
    @parametrize("worker_prefetch_cnt", [0, 5, 10])
    def test_worker_exception_raised(self, num_workers, worker_prefetch_cnt):
        dp = IterableWrapper(range(100)).sharding_filter()
        dp = MakeMistakeDataPipe(dp)
        rs = MultiProcessingReadingService(num_workers=num_workers, worker_prefetch_cnt=worker_prefetch_cnt)
        dl = DataLoader2(dp, reading_service=rs)
        it = iter(dl)
        for _ in range(EXCEPTION_ITERATION_NUM * num_workers):
            next(it)
        with self.assertRaises(_CustomException) as cm:
            next(it)
        exc_msg = str(cm.exception)
        self.assertTrue("Caught _CustomException in worker process 0" in exc_msg)
        self.assertTrue("Original Traceback" in exc_msg)
        self.assertTrue("_CustomException: oops" in exc_msg)

    @parametrize("num_workers", [1, 3])
    @parametrize("worker_prefetch_cnt", [0, 5, 10])
    def test_dispatching_exception_raised(self, num_workers, worker_prefetch_cnt):
        dp = IterableWrapper(range(100))
        dp = MakeMistakeDataPipe(dp)
        dp = dp.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
        dp = dp.map(_x_mult_2)
        rs = MultiProcessingReadingService(num_workers=num_workers, worker_prefetch_cnt=worker_prefetch_cnt)
        dl = DataLoader2(dp, reading_service=rs)
        it = iter(dl)
        for _ in range(EXCEPTION_ITERATION_NUM):
            next(it)
        with self.assertRaises(_CustomException) as cm:
            next(it)
        exc_msg = str(cm.exception)
        self.assertTrue("Caught _CustomException in dispatching process" in exc_msg)
        self.assertTrue("Original Traceback" in exc_msg)
        self.assertTrue("_CustomException: oops" in exc_msg)


TEST_MASTER_ADDR = "127.0.0.1"
DEFAULT_WORLD_SIZE = 2


def _get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


class TerminateSignal:
    pass


def _launch_distributed_training(world_size, *args, fn):
    os.environ["MASTER_ADDR"] = TEST_MASTER_ADDR
    os.environ["MASTER_PORT"] = _get_open_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = []
    for rank in range(world_size):
        p = ctx.Process(
            target=fn,
            args=(
                rank,
                world_size,
                q,
                *args,
            ),
        )
        p.start()
        ps.append(p)
    res = []
    while True:
        try:
            d = q.get()
            if isinstance(d, TerminateSignal):
                break
            res.append(d)
        except queue.Empty:
            continue
    for p in ps:
        p.join()
    return res


def _dist_one_epoch(dl):
    res = []
    for d in dl:
        res.append(d)
        # Simulate training synchronization
        dist.barrier()
    return res


def _finalize_distributed_queue(rank, q):
    r"""
    Synchronize all distributed processes to guarantee all data have been put into
    the Multiprocessing Queue.
    """
    pg = dist.new_group(backend="gloo")
    end_tensor = torch.tensor([rank], dtype=torch.int64)
    dist.all_reduce(end_tensor, group=pg)
    if rank == 0:
        q.put(TerminateSignal())

    dist.destroy_process_group(pg)


def _random_fn(data):
    r"""
    Used to validate the randomness of subprocess-local RNGs are set deterministically.
    """
    py_random_num = random.randint(0, 2 ** 32)
    np_random_num = np.random.randint(0, 2 ** 32)
    torch_random_num = torch.randint(0, 2 ** 32, size=[]).item()
    return (data, py_random_num, np_random_num, torch_random_num)


def _dist_training_fn(rank, world_size, q, dp_fn, rs_fn, num_workers, ctx):
    # Use gloo
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Uneven shards
    data_length = world_size * num_workers * 10 + 1
    dp = dp_fn(data_length)
    rs = rs_fn(num_workers, ctx)
    dl = DataLoader2(dp, reading_service=rs)

    # No seed
    res = _dist_one_epoch(dl)
    q.put((0, rank, res))

    # Shuffle with seed
    for epoch in range(2):
        dl.seed(123)
        res = _dist_one_epoch(dl)
        q.put((epoch + 1, rank, res))

    # Different seed
    dl.seed(321)
    res = _dist_one_epoch(dl)
    q.put((3, rank, res))

    _finalize_distributed_queue(rank, q)

    dl.shutdown()


@skipIfNoDistributed
@unittest.skipIf(IS_WINDOWS, "Remove when https://github.com/pytorch/data/issues/857 is fixed")
class SequentialReadingServiceTest(TestCase):
    @staticmethod
    def _make_dp(data_length):
        data_source = IterableWrapper(list(range(data_length)))
        dp = data_source.shuffle().sharding_filter().map(_random_fn)
        return dp

    @staticmethod
    def _make_dispatching_dp(data_length):
        data_source = IterableWrapper(list(range(data_length)))
        dp = data_source.shuffle().sharding_filter()
        dp = dp.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING).map(_random_fn)
        return dp

    @staticmethod
    def _make_rs(num_workers, ctx):
        mp_rs = MultiProcessingReadingService(
            num_workers=num_workers,
            multiprocessing_context=ctx,
        )
        dist_rs = DistributedReadingService()
        rs = SequentialReadingService(dist_rs, mp_rs)
        return rs

    @mp_ctx_parametrize
    def test_sequential_reading_service_normal_dp(self, ctx):
        world_size = DEFAULT_WORLD_SIZE
        num_workers = 2
        res = _launch_distributed_training(
            world_size,
            SequentialReadingServiceTest._make_dp,
            SequentialReadingServiceTest._make_rs,
            num_workers,
            ctx,
            fn=_dist_training_fn,
        )
        result = ({}, {}, {}, {})
        for epoch, rank, r in res:
            d, *ran_nums = list(zip(*r))
            result[epoch][rank] = (d, ran_nums)

        # Guarantee the same length per rank
        for rr in result:
            exp_len = num_workers * 10
            for _, (d, _) in rr.items():
                self.assertEqual(len(d), exp_len)

        # Same seed generate the same order of data and the same random state
        self.assertEqual(result[1], result[2])

        # Different seeds
        for rank in range(world_size):
            # Different shuffle order
            self.assertNotEqual(result[1][rank][0], result[3][rank][0])
            # Different subprocess-local random state
            self.assertNotEqual(result[1][rank][1], result[3][rank][1])

    @mp_ctx_parametrize
    def test_sequential_reading_service_dispatching_dp(self, ctx):
        world_size = DEFAULT_WORLD_SIZE
        num_workers = 2
        res = _launch_distributed_training(
            world_size,
            SequentialReadingServiceTest._make_dispatching_dp,
            SequentialReadingServiceTest._make_rs,
            num_workers,
            ctx,
            fn=_dist_training_fn,
        )
        result = ({}, {}, {}, {})
        for epoch, rank, r in res:
            d, *ran_nums = list(zip(*r))
            result[epoch][rank] = (d, ran_nums)

        # Guarantee the same length per rank
        for rr in result:
            exp_len = num_workers * 10
            for _, (d, _) in rr.items():
                self.assertEqual(len(d), exp_len)

        # Same seed generate the same order of data and the same random state
        self.assertEqual(result[1], result[2])

        # Different seeds
        for rank in range(world_size):
            # Different shuffle order
            self.assertNotEqual(result[1][rank][0], result[3][rank][0])
            # Different subprocess-local random state
            self.assertNotEqual(result[1][rank][1], result[3][rank][1])


instantiate_parametrized_tests(MultiProcessingReadingServiceTest)
instantiate_parametrized_tests(SequentialReadingServiceTest)


if __name__ == "__main__":
    unittest.main()
