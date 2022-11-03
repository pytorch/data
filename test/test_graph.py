# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Dict, Iterator, List, Tuple, TypeVar

import expecttest

from _utils._common_utils_for_test import IS_WINDOWS
from torch.utils.data import IterDataPipe
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, ReadingServiceInterface
from torchdata.dataloader2.graph import find_dps, remove_dp, replace_dp, traverse_dps
from torchdata.datapipes.iter import IterableWrapper, Mapper
from torchdata.datapipes.utils import to_graph

T_co = TypeVar("T_co", covariant=True)

try:
    import graphviz

    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False


class Adaptor(IterDataPipe[T_co]):
    def __init__(self, datapipe: IterDataPipe) -> None:
        self.datapipe = datapipe
        self.started = False

    def __iter__(self) -> Iterator[T_co]:
        yield from self.datapipe


class DummyIterDataPipe(IterDataPipe[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        yield from range(10)


class TempReadingService(ReadingServiceInterface):
    adaptors: List[IterDataPipe] = []

    def initialize(self, datapipe: IterDataPipe) -> IterDataPipe:
        graph = traverse_dps(datapipe)
        dps = find_dps(graph, Mapper)

        for dp in reversed(dps):
            new_dp = Adaptor(dp)
            self.adaptors.append(new_dp)
            graph = replace_dp(graph, dp, new_dp)

        return list(graph.values())[0][0]

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

        return traverse_dps(dp), (src_dp, m1, ub, dm, c1, c2, m2, dp)

    def test_find_dps(self) -> None:
        graph, (_, m1, *_, m2, _) = self._get_datapipes()  # pyre-ignore

        dps = find_dps(graph, Mapper)

        expected_dps = {m1, m2}
        for dp in dps:
            self.assertTrue(dp in expected_dps)

    def _validate_graph(self, graph, nested_dp):
        self.assertEqual(len(graph), len(nested_dp))
        for dp_id, sub_nested_dp in zip(graph, nested_dp):
            self.assertEqual(graph[dp_id][0], sub_nested_dp[0])
            if len(graph[dp_id][1]) > 0:
                self._validate_graph(graph[dp_id][1], sub_nested_dp[1])

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
        new_dp3 = DummyIterDataPipe()

        graph = replace_dp(graph, m1, new_dp1)
        exp_g1 = [
            [
                dp,
                [
                    [m2, [[c1, [[dm, [[ub, [[new_dp1, [[m1, [[src_dp, []]]]]]]]]]]]]],
                    [c2, [[dm, [[ub, [[new_dp1, [[m1, [[src_dp, []]]]]]]]]]]],
                ],
            ]
        ]
        self._validate_graph(traverse_dps(dp), exp_g1)

        graph = replace_dp(graph, m2, new_dp2)
        exp_g2 = [
            [
                dp,
                [
                    [new_dp2, [[m2, [[c1, [[dm, [[ub, [[new_dp1, [[m1, [[src_dp, []]]]]]]]]]]]]]]],
                    [c2, [[dm, [[ub, [[new_dp1, [[m1, [[src_dp, []]]]]]]]]]]],
                ],
            ]
        ]
        self._validate_graph(traverse_dps(dp), exp_g2)

        graph = replace_dp(graph, m1, new_dp3)
        exp_g3 = [
            [
                dp,
                [
                    [new_dp2, [[m2, [[c1, [[dm, [[ub, [[new_dp1, [[new_dp3, []]]]]]]]]]]]]],
                    [c2, [[dm, [[ub, [[new_dp1, [[new_dp3, []]]]]]]]]],
                ],
            ]
        ]
        self._validate_graph(traverse_dps(dp), exp_g3)

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
        exp_g1 = [[dp, [[m2, [[c1, [[dm, [[ub, [[src_dp, []]]]]]]]]], [c2, [[dm, [[ub, [[src_dp, []]]]]]]]]]]
        self._validate_graph(traverse_dps(dp), exp_g1)

        graph = remove_dp(graph, m2)
        exp_g2 = [[dp, [[c1, [[dm, [[ub, [[src_dp, []]]]]]]], [c2, [[dm, [[ub, [[src_dp, []]]]]]]]]]]
        self._validate_graph(traverse_dps(dp), exp_g2)

        with self.assertRaisesRegex(RuntimeError, "Cannot remove the source DataPipe"):
            remove_dp(graph, src_dp)

        with self.assertRaisesRegex(RuntimeError, "Cannot remove the receiving DataPipe"):
            remove_dp(graph, dp)

    def test_reading_service(self) -> None:
        _, (*_, dp) = self._get_datapipes()  # pyre-ignore

        rs = TempReadingService()
        dl = DataLoader2(dp, reading_service=rs)

        self.assertTrue(len(rs.adaptors) == 0)

        it = iter(dl)
        for new_dp in rs.adaptors:
            self.assertTrue(new_dp.started)

        res = list(it)
        self.assertEqual(len(res), 20)

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


class TestGraphVisualization(expecttest.TestCase):
    @unittest.skipIf(not HAS_GRAPHVIZ, "Package `graphviz` is required to test graph visualization functionalities.")
    def test_to_graph(self):
        dp1 = IterableWrapper(range(10))
        dp2 = dp1.map(lambda x: x + 1)
        dp3 = dp2.filter(lambda x: x > 5)
        cdp1, cdp2 = dp3.fork(num_instances=2)
        dp4 = cdp1.zip(cdp2)
        cdp3, cdp4 = dp4.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        dp5 = cdp3.concat(cdp4)

        # Test to ensure that we can create these graphs with runtime errors
        kwargs_list: List[Dict] = [
            {"dp": dp1},
            {"dp": dp2},
            {"dp": dp3},
            {"dp": cdp1, "debug": True},
            {"dp": dp4},
            {"dp": dp4, "debug": True},
            {"dp": cdp3, "debug": True},
            {"dp": dp5},
            {"dp": dp5, "debug": True},
        ]
        for kwargs in kwargs_list:
            g = to_graph(**kwargs)
            self.assertTrue(isinstance(g, graphviz.Digraph))


if __name__ == "__main__":
    unittest.main()
