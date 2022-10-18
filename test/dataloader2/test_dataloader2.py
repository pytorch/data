# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pickle
import unittest
from unittest import TestCase

from torchdata.dataloader2 import (
    DataLoader2,
    MultiProcessingReadingService,
    PrototypeMultiProcessingReadingService,
    ReadingServiceInterface,
)
from torchdata.dataloader2.dataloader2 import READING_SERVICE_STATE_KEY_NAME, SERIALIZED_DATAPIPE_KEY_NAME

from torchdata.dataloader2.graph import DataPipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe


def _filter_fn(x: int):
    return x < 5


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

    def test_dataloader2_len(self) -> None:

        reading_services = (
            None,
            MultiProcessingReadingService(num_workers=2),
            PrototypeMultiProcessingReadingService(num_workers=2),
        )

        sharding_dp = IterableWrapper(range(10)).sharding_filter()
        filter_dp = IterableWrapper(range(10)).filter(_filter_fn)

        for rs in reading_services:
            # Functional Test: Case with sharding
            data_loader_sharding: DataLoader2 = DataLoader2(datapipe=sharding_dp, reading_service=rs)
            # Call `__len__` before initialization
            self.assertEqual(10, len(data_loader_sharding))
            _it = iter(data_loader_sharding)
            # Length should stay the same after calling `__iter__`
            self.assertEqual(10, len(data_loader_sharding))
            list(_it)
            self.assertEqual(10, len(data_loader_sharding))

            # Functional Test: Case with filter
            data_loader_filter: DataLoader2 = DataLoader2(datapipe=filter_dp, reading_service=rs)
            with self.assertRaisesRegex(RuntimeError, "Unable to retrieve the length of the DataPipe"):
                print(len(data_loader_filter))

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

        test_data_pipe_2 = IterableWrapper(range(5))
        restored_data_loader: DataLoader2 = DataLoader2(datapipe=test_data_pipe_2, reading_service=reading_service)
        restored_data_loader.load_state_dict(state)

        restored_data_loader_datapipe = restored_data_loader.datapipe
        deserialized_datapipe = pickle.loads(state[SERIALIZED_DATAPIPE_KEY_NAME])
        for batch_1, batch_2 in zip(restored_data_loader_datapipe, deserialized_datapipe):
            self.assertEqual(batch_1, batch_2)

        self.assertNotEqual(
            len(restored_data_loader.datapipe),
            len(test_data_pipe_2),
        )

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
            PrototypeMultiProcessingReadingService(num_workers=4, prefetch_worker=0),
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
            self._get_mp_reading_service,
            self._get_proto_reading_service,
            self._get_mp_reading_service_zero_workers,
            self._get_proto_reading_service_zero_workers,
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


if __name__ == "__main__":
    unittest.main()
