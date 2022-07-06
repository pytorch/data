# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Iterator, List, Tuple, TypeVar

import expecttest

from _utils._common_utils_for_test import IS_WINDOWS
from torch.utils.data import IterDataPipe
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, ReadingServiceInterface
from torchdata.dataloader2.graph import find_dps, remove_dp, replace_dp, traverse
from torchdata.datapipes.iter import IterableWrapper, Mapper

T_co = TypeVar("T_co", covariant=True)


class Adaptor(IterDataPipe[T_co]):
    def __init__(self, datapipe: IterDataPipe) -> None:
        self.datapipe = datapipe
        self.started = False

    def __iter__(self) -> Iterator[T_co]:
        yield from self.datapipe


class TempReadingService(ReadingServiceInterface):
    adaptors: List[IterDataPipe] = []

    def initialize(self, datapipe: IterDataPipe) -> IterDataPipe:
        graph = traverse(datapipe, only_datapipe=True)
        dps = find_dps(graph, Mapper)

        for dp in reversed(dps):
            new_dp = Adaptor(dp)
            self.adaptors.append(new_dp)
            graph = replace_dp(graph, dp, new_dp)

        return list(graph.keys())[0]

    def initialize_iteration(self) -> None:
        for dp in self.adaptors:
            dp.started = True

    def finalize_iteration(self) -> None:
        for dp in self.adaptors:
            dp.started = False


def _x_and_x_plus_5(x):
    return [x, x + 5]


def _x_mod_2(x):
    return x % 2


def _x_mult_2(x):
    return x * 2


class TestGraph(expecttest.TestCase):
    def _get_datapipes(self) -> Tuple[IterDataPipe, IterDataPipe, IterDataPipe]:
        src_dp = IterableWrapper(range(20))
        m1 = src_dp.map(_x_and_x_plus_5)
        ub = m1.unbatch()
        c1, c2 = ub.demux(2, _x_mod_2)
        dm = c1.main_datapipe
        m2 = c1.map(_x_mult_2)
        dp = m2.zip(c2)

        return traverse(dp, only_datapipe=True), (src_dp, m1, ub, dm, c1, c2, m2, dp)

    def test_find_dps(self) -> None:
        graph, (_, m1, *_, m2, _) = self._get_datapipes()  # pyre-ignore

        dps = find_dps(graph, Mapper)

        expected_dps = {m1, m2}
        for dp in dps:
            self.assertTrue(dp in expected_dps)

    def test_replace_dps(self) -> None:
        # pyre-fixme[23]: Unable to unpack 3 values, 2 were expected.
        graph, (
            src_dp,
            m1,
            ub,
            dm,
            c1,
            c2,
            m2,
            dp,
        ) = self._get_datapipes()

        new_dp1 = Adaptor(m1)
        new_dp2 = Adaptor(m2)

        graph = replace_dp(graph, m1, new_dp1)
        exp_g1 = {
            dp: {
                m2: {c1: {dm: {ub: {new_dp1: {m1: {src_dp: {}}}}}}},
                c2: {dm: {ub: {new_dp1: {m1: {src_dp: {}}}}}},
            }
        }
        self.assertEqual(graph, exp_g1)
        self.assertEqual(traverse(dp, only_datapipe=True), exp_g1)

        graph = replace_dp(graph, m2, new_dp2)
        exp_g2 = {
            dp: {
                new_dp2: {m2: {c1: {dm: {ub: {new_dp1: {m1: {src_dp: {}}}}}}}},
                c2: {dm: {ub: {new_dp1: {m1: {src_dp: {}}}}}},
            }
        }
        self.assertEqual(graph, exp_g2)
        self.assertEqual(traverse(dp, only_datapipe=True), exp_g2)

    def test_remove_dps(self) -> None:
        # pyre-fixme[23]: Unable to unpack 3 values, 2 were expected.
        graph, (
            src_dp,
            m1,
            ub,
            dm,
            c1,
            c2,
            m2,
            dp,
        ) = self._get_datapipes()

        graph = remove_dp(graph, m1)
        exp_g1 = {dp: {m2: {c1: {dm: {ub: {src_dp: {}}}}}, c2: {dm: {ub: {src_dp: {}}}}}}
        self.assertEqual(graph, exp_g1)
        self.assertEqual(traverse(dp, only_datapipe=True), exp_g1)

        graph = remove_dp(graph, m2)
        exp_g2 = {dp: {c1: {dm: {ub: {src_dp: {}}}}, c2: {dm: {ub: {src_dp: {}}}}}}
        self.assertEqual(graph, exp_g2)
        self.assertEqual(traverse(dp, only_datapipe=True), exp_g2)

        with self.assertRaisesRegex(
            Exception,
            "Cannot remove source DataPipe that is the first DataPipe in the pipeline",
        ):
            remove_dp(graph, src_dp)

        with self.assertRaisesRegex(
            Exception,
            "Cannot remove a receiving DataPipe having multiple sending DataPipes",
        ):
            remove_dp(graph, dp)

    def test_reading_service(self) -> None:
        _, (*_, dp) = self._get_datapipes()  # pyre-ignore

        rs = TempReadingService()
        dl = DataLoader2(dp, reading_service=rs)

        self.assertTrue(len(rs.adaptors) == 0)

        it = iter(dl)
        for new_dp in rs.adaptors:
            self.assertTrue(new_dp.started)

        _ = list(it)

        for new_dp in rs.adaptors:
            self.assertFalse(new_dp.started)

    @unittest.skipIf(IS_WINDOWS, "Fork is required for lambda")
    def test_multiprocessing_reading_service(self) -> None:
        _, (*_, dp) = self._get_datapipes()  # pyre-ignore
        rs = MultiProcessingReadingService(2, persistent_workers=True, multiprocessing_context="fork")
        dl = DataLoader2(dp, reading_service=rs)
        d1 = list(dl)
        d2 = list(dl)
        self.assertEqual(d1, d2)


if __name__ == "__main__":
    unittest.main()
