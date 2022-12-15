# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle
import unittest
from unittest import TestCase

import torch

from torch.utils.data.datapipes.iter.grouping import SHARDING_PRIORITIES

from torchdata.dataloader2 import (
    communication,
    DataLoader2,
    MultiProcessingReadingService,
    PrototypeMultiProcessingReadingService,
    ReadingServiceInterface,
)
from torchdata.dataloader2.dataloader2 import READING_SERVICE_STATE_KEY_NAME, SERIALIZED_DATAPIPE_KEY_NAME

from torchdata.dataloader2.graph import DataPipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
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
TEST_WITH_TSAN = os.getenv("PYTORCH_TEST_WITH_TSAN", "0") == "1"


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

    def test_dataloader2_multi_process_reading_service(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        reading_service = MultiProcessingReadingService()
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
            PrototypeMultiProcessingReadingService(num_workers=4, worker_prefetch_cnt=0),
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
    def _get_proto_reading_service():
        return PrototypeMultiProcessingReadingService(num_workers=2)

    @staticmethod
    def _get_mp_reading_service_zero_workers():
        return MultiProcessingReadingService(num_workers=0)

    @staticmethod
    def _get_proto_reading_service_zero_workers():
        return PrototypeMultiProcessingReadingService(num_workers=0)

    # def _collect_data(self, datapipe, reading_service_gen):
    #     dl: DataLoader2 = DataLoader2(datapipe, reading_service=reading_service_gen())
    #     result = []
    #     # Testing how RS handles partial reading and reiterations
    #     for row, _ in zip(dl, range(10)):
    #         result.append(row)
    #     print("Attempts to shutdown")
    #     dl.shutdown()
    #     dl: DataLoader2 = DataLoader2(datapipe, reading_service=reading_service_gen())
    #     print(f"half way through: {result = }", flush=True)
    #     for row in dl:
    #         result.append(row)
    #     print(f"full way through: {result = }", flush=True)
    #     return result

    # Original
    def _collect_data(self, datapipe, reading_service_gen):
        dl: DataLoader2 = DataLoader2(datapipe, reading_service=reading_service_gen())
        result = []
        # Testing how RS handles partial reading and reiterations
        for row, _ in zip(dl, range(10)):
            result.append(row)
        for row in dl:
            result.append(row)
        return result

    @staticmethod
    def _no_op(x):
        return x

    def test_dataloader2_batch_collate(self) -> None:
        dp: IterDataPipe = IterableWrapper(range(100)).batch(2).sharding_filter().collate(self._no_op)  # type: ignore[assignment]
        expected = self._collect_data(dp, reading_service_gen=self._get_no_reading_service)

        reading_service_generators = (
            # self._get_mp_reading_service,
            self._get_proto_reading_service,
            # self._get_mp_reading_service_zero_workers,
            # self._get_proto_reading_service_zero_workers,
        )
        for reading_service_gen in reading_service_generators:
            actual = self._collect_data(dp, reading_service_gen=reading_service_gen)
            # TODO(588): This comparison only indicates that somethings is broken and not helping with debug
            self.assertEqual(expected, actual, reading_service_gen)

    def test_dataloader2_shuffle(self) -> None:
        # TODO(589): Add shuffle test
        pass


class DataLoader2IntegrationTest(TestCase):
    @staticmethod
    def _get_mp_reading_service():
        return MultiProcessingReadingService(num_workers=2)

    @staticmethod
    def _access_datapipe(dl):
        """
        Returns a reference to the DataPipe, bypassing serialization wrapper and etc.
        """
        return dl.datapipe._datapipe

    def test_lazy_load(self):
        source_dp = IterableWrapper([(i, i) for i in range(10)])
        map_dp = source_dp.to_map_datapipe()

        reading_service_generators = (self._get_mp_reading_service,)
        for reading_service_gen in reading_service_generators:
            dl: DataLoader2 = DataLoader2(datapipe=map_dp, reading_service=reading_service_gen())
            # Lazy loading
            dp = self._access_datapipe(dl)
            self.assertTrue(dp._map is None)
            it = iter(dl)
            self.assertEqual(list(it), list(range(10)))
            # Lazy loading in multiprocessing
            self.assertTrue(map_dp._map is None)


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
    #     (process, req_queue, res_queue, _thread_local_datapipe) = communication.eventloop.SpawnThreadForDataPipeline(numbers_dp)
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
        (process, req_queue, res_queue, _thread_local_datapipe) = communication.eventloop.SpawnThreadForDataPipeline(
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


class PrototypeMultiProcessingReadingServiceTest(TestCase):
    @staticmethod
    def _worker_init_fn(datapipe, worker_info):
        datapipe = datapipe.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(
            datapipe, worker_info.num_workers, worker_info.worker_id, SHARDING_PRIORITIES.MULTIPROCESSING
        )
        return datapipe

    @staticmethod
    def _worker_reset_fn(datapipe, worker_info):
        worker_seed_generator = torch.Generator()
        worker_seed_generator.manual_seed(123)
        torch.utils.data.graph_settings.apply_random_seed(
            datapipe,
            worker_seed_generator,
        )
        return datapipe

    def test_worker_fns(self):
        dp: IterDataPipe = IterableWrapper(range(100)).batch(2).shuffle()
        torch.manual_seed(123)
        exp = list(dp)

        rs = PrototypeMultiProcessingReadingService(
            num_workers=1, worker_init_fn=self._worker_init_fn, worker_reset_fn=self._worker_reset_fn
        )
        dl = DataLoader2(dp, reading_service=rs)

        # Test worker_init_fn to shard the DataPipe graph
        res1 = list(dl)
        self.assertEqual(exp, res1)

        # Test worker_reset_fn to set the same random seed across epoches
        res2 = list(dl)
        self.assertEqual(exp, res2)


if __name__ == "__main__":
    unittest.main()
