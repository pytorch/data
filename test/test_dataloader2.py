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
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe


class TestReadingService(ReadingServiceInterface):
    def initialize(self, dp: IterDataPipe) -> IterDataPipe:
        return dp


class DataLoader2Test(TestCase):
    def test_dataloader2(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        data_loader = DataLoader2(datapipe=test_data_pipe)

        expected_batch = 0
        for batch in iter(data_loader):
            self.assertEqual(batch, expected_batch)
            expected_batch += 1

    def test_dataloader2_shutdown(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        data_loader = DataLoader2(datapipe=test_data_pipe)
        data_loader.shutdown()

    def test_dataloader2_state_dict(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        data_loader = DataLoader2(datapipe=test_data_pipe)

        state = data_loader.state_dict()
        self.assertIsNotNone(state)
        self.assertIsNotNone(state[SERIALIZED_DATAPIPE_KEY_NAME])
        self.assertIsNone(state[READING_SERVICE_STATE_KEY_NAME])
        data_loader.shutdown()

    def test_dataloader2_reading_service(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        reading_service = TestReadingService()
        data_loader = DataLoader2(datapipe=test_data_pipe, reading_service=reading_service)

        expected_batch = 0
        for batch in iter(data_loader):
            self.assertEqual(batch, expected_batch)
            expected_batch += 1

    def test_dataloader2_multi_process_reading_service(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        reading_service = MultiProcessingReadingService()
        data_loader = DataLoader2(datapipe=test_data_pipe, reading_service=reading_service)

        expected_batch = 0
        for batch in iter(data_loader):
            self.assertEqual(batch, expected_batch)
            expected_batch += 1

    def test_dataloader2_load_state_dict(self) -> None:
        test_data_pipe = IterableWrapper(range(3))
        reading_service = TestReadingService()
        data_loader = DataLoader2(datapipe=test_data_pipe, reading_service=reading_service)

        batch = next(iter(data_loader))
        self.assertEqual(batch, 0)

        state = data_loader.state_dict()
        self.assertIsNotNone(state)
        self.assertIsNotNone(state[SERIALIZED_DATAPIPE_KEY_NAME])
        self.assertIsNone(state[READING_SERVICE_STATE_KEY_NAME])
        data_loader.shutdown()

        test_data_pipe_2 = IterableWrapper(range(5))
        restored_data_loader = DataLoader2(datapipe=test_data_pipe_2, reading_service=reading_service)
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


if __name__ == "__main__":
    unittest.main()
