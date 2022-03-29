# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import expecttest
from torchdata.datapipes.map import MapDataPipe, SequenceWrapper, UnZipper


class TestMapDataPipe(expecttest.TestCase):
    def test_unzipper_mapdatapipe(self) -> None:
        source_dp = SequenceWrapper([(i, i + 10, i + 20) for i in range(10)])

        # Functional Test: unzips each sequence, no `sequence_length` specified
        dp1: MapDataPipe
        dp2: MapDataPipe
        dp3: MapDataPipe
        dp1, dp2, dp3 = UnZipper(source_dp, sequence_length=3)  # type: ignore[misc]
        self.assertEqual(list(range(10)), list(dp1))
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        # Functional Test: unzips each sequence, with `sequence_length` specified
        dp1, dp2, dp3 = source_dp.unzip(sequence_length=3)
        self.assertEqual(list(range(10)), list(dp1))
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        # Functional Test: skipping over specified values
        dp2, dp3 = source_dp.unzip(sequence_length=3, columns_to_skip=[0])
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        (dp2,) = source_dp.unzip(sequence_length=3, columns_to_skip=[0, 2])
        self.assertEqual(list(range(10, 20)), list(dp2))

        source_dp = SequenceWrapper([(i, i + 10, i + 20, i + 30) for i in range(10)])
        dp2, dp3 = source_dp.unzip(sequence_length=4, columns_to_skip=[0, 3])
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        # __len__ Test: the lengths of child DataPipes are correct
        self.assertEqual((10, 10), (len(dp2), len(dp3)))


if __name__ == "__main__":
    unittest.main()
