# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import expecttest
from torchdata.datapipes.iter import MapToIterConverter
from torchdata.datapipes.map import InMemoryCacheHolder, MapDataPipe, SequenceWrapper, UnZipper


class TestMapDataPipe(expecttest.TestCase):
    def test_unzipper_mapdatapipe(self) -> None:
        source_dp = SequenceWrapper([(i, i + 10, i + 20) for i in range(10)])

        # Functional Test: unzips each sequence, with `sequence_length` specified
        dp1: MapDataPipe
        dp2: MapDataPipe
        dp3: MapDataPipe
        dp1, dp2, dp3 = UnZipper(source_dp, sequence_length=3)  # type: ignore[misc]
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

    def test_map_to_iter_converter_datapipe(self) -> None:
        # Functional Test: ensure the conversion without indices input is correct
        source_dp = SequenceWrapper(range(10))
        iter_dp = source_dp.to_iter_datapipe()
        self.assertEqual(list(range(10)), list(iter_dp))

        # Functional Test: ensure conversion with custom indices is correct
        source_dp2 = SequenceWrapper({"a": 0, "b": 1, "c": 2})
        iter_dp2 = MapToIterConverter(source_dp2, indices=["a", "b", "c"])
        self.assertEqual([0, 1, 2], list(iter_dp2))

        # __len__ Test: the lengths of the output is correct
        self.assertEqual(10, len(iter_dp))
        self.assertEqual(3, len(iter_dp2))

    def test_in_memory_cache_holder_mapdatapipe(self) -> None:
        source_dp = SequenceWrapper(range(10))
        cache_dp = source_dp.in_memory_cache()

        # Functional Test: Cache DP should just return the data without changing the values
        self.assertEqual(list(range(10)), list(cache_dp))

        # Functional Test: Ensure the objects are the same ones from source DataPipe
        cache_dp = InMemoryCacheHolder(source_dp)  # type: ignore[arg-type]
        res1 = list(cache_dp)
        res2 = list(cache_dp)
        self.assertTrue(id(source) == id(cache) for source, cache in zip(source_dp, res1))
        self.assertTrue(id(source) == id(cache) for source, cache in zip(source_dp, res2))

        # __len__ Test: inherits length from source_dp
        self.assertEqual(10, len(cache_dp))


if __name__ == "__main__":
    unittest.main()
