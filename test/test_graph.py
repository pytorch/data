# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

from typing import Dict, Iterator, List, Tuple, TypeVar

import expecttest

from _utils._common_utils_for_test import IS_WINDOWS

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, ReadingServiceInterface
from torchdata.dataloader2.graph import find_dps, list_dps, remove_dp, replace_dp, traverse_dps
from torchdata.dataloader2.graph.utils import _find_replicable_branches
from torchdata.dataloader2.random import SeedGenerator
from torchdata.dataloader2.utils.dispatch import (
    _DummyIterDataPipe,
    find_lca_round_robin_sharding_dp,
    find_non_dispatching_branches,
)
from torchdata.datapipes.iter import IterableWrapper, Mapper, ShardingRoundRobinDispatcher
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

    def initialize_iteration(self, seed_generator: SeedGenerator) -> None:
        seed_generator.seed(123)
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

    def test_list_dps(self) -> None:
        def _validate_fn(dps, exp_dps):
            self.assertEqual(len(dps), len(exp_dps))
            # Validate BFS Order
            for dp, exp_dp in zip(dps, exp_dps):
                self.assertEqual(dp, exp_dp)

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
        exp_all_dps = [dp, m2, c2, c1, dm, ub, m1, src_dp]

        # List all DataPipes
        dps = list_dps(graph)
        _validate_fn(dps, exp_all_dps)

        # List all DataPipes excluding a single DataPipe
        dps = list_dps(graph, exclude_dps=m1)
        *exp_dps, _, _ = exp_all_dps
        _validate_fn(dps, exp_dps)

        # Exclude a DataPipe on one branch
        dps = list_dps(graph, exclude_dps=m2)
        exp_dps = [dp, c2]
        _validate_fn(dps, exp_dps)

        # List all DataPipes excluding multiple DataPipes
        dps = list_dps(graph, exclude_dps=[m1, m2])
        exp_dps = [dp, c2]
        _validate_fn(dps, exp_dps)

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

        self.assertEqual(res, list(dl))

    @unittest.skipIf(IS_WINDOWS, "Fork is required for lambda")
    def test_multiprocessing_reading_service(self) -> None:
        _, (*_, dp) = self._get_datapipes()  # pyre-ignore
        rs = MultiProcessingReadingService(2, persistent_workers=True, multiprocessing_context="fork")
        dl = DataLoader2(dp, reading_service=rs)
        d1 = list(dl)
        d2 = list(dl)
        self.assertEqual(d1, d2)


def insert_round_robin_sharding(graph, datapipe):
    dispatch_dp = ShardingRoundRobinDispatcher(datapipe, SHARDING_PRIORITIES.MULTIPROCESSING)
    return replace_dp(graph, datapipe, dispatch_dp), dispatch_dp


def replace_by_dummy(graph, datapipe):
    return replace_dp(graph, datapipe, _DummyIterDataPipe())


def make_non_replicable_dp(datapipe):
    datapipe.is_replicable = types.MethodType(lambda self: False, datapipe)
    return datapipe


class TestNonReplicableDataPipe(expecttest.TestCase):
    def _make_dp(self):
        r"""
        Create a DataPipe that contains the most of cases including:
            - single-branch pipeline
            - multi-branch pipeline
            - pipeline that has circurlar references

        single_br_dp -------------------------------------
                                     ch1                    \
                                   /     \                   \
        multi_br_dp -->forker_dp-->       -> fork_zip_dp -> end_dp ->
                                   \     /                   /
               <-------              ch2                    /
             /          \                                  /
        cir_br_dp -> cir_map_dp --------------------------
        """
        # Single-branch
        single_br_dp = IterableWrapper(list(range(10)))

        # Multi-branch
        multi_br_dp = IterableWrapper(list(range(10)))
        ch1, ch2 = multi_br_dp.fork(2)
        forker_dp = ch1.main_datapipe
        fork_zip_dp = ch1.zip(ch2)

        # Circular-branch
        cir_br_dp = IterableWrapper(list(range(10)))
        cir_map_dp = cir_br_dp.map(_x_mult_2)
        # Force to circular reference
        cir_br_dp.cir_dep = cir_map_dp

        end_dp = single_br_dp.zip(fork_zip_dp, cir_map_dp)
        graph = traverse_dps(end_dp)
        return single_br_dp, multi_br_dp, forker_dp, ch1, ch2, fork_zip_dp, cir_br_dp, cir_map_dp, end_dp, graph

    def test_single_round_robin_sharding_dp(self):
        single_br_dp, *_, graph = self._make_dp()
        graph, single_br_dp = insert_round_robin_sharding(graph, single_br_dp)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), single_br_dp)

        # The same non-shardable DataPipe on both branches
        _, multi_br_dp, *_, graph = self._make_dp()
        graph, multi_br_dp = insert_round_robin_sharding(graph, multi_br_dp)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), multi_br_dp)

        _, _, _, ch1, _, fork_zip_dp, *_, graph = self._make_dp()
        graph, ch1 = insert_round_robin_sharding(graph, ch1)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), fork_zip_dp)

        # Circular reference
        *_, cir_br_dp, cir_map_dp, _, graph = self._make_dp()
        graph, cir_br_dp = insert_round_robin_sharding(graph, cir_br_dp)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), cir_map_dp)

        *_, cir_map_dp, _, graph = self._make_dp()
        graph, cir_map_dp = insert_round_robin_sharding(graph, cir_map_dp)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), cir_map_dp)

    def test_multi_round_robin_sharding_dps(self):
        single_br_dp, multi_br_dp, *_, end_dp, graph = self._make_dp()
        graph, single_br_dp = insert_round_robin_sharding(graph, single_br_dp)
        graph, multi_br_dp = insert_round_robin_sharding(graph, multi_br_dp)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), end_dp)

        single_br_dp, _, _, ch1, *_, end_dp, graph = self._make_dp()
        graph, single_br_dp = insert_round_robin_sharding(graph, single_br_dp)
        graph, ch1 = insert_round_robin_sharding(graph, ch1)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), end_dp)

        _, multi_br_dp, _, ch1, _, fork_zip_dp, *_, graph = self._make_dp()
        graph, multi_br_dp = insert_round_robin_sharding(graph, multi_br_dp)
        graph, ch1 = insert_round_robin_sharding(graph, ch1)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), fork_zip_dp)

        single_br_dp, *_, cir_br_dp, _, end_dp, graph = self._make_dp()
        graph, single_br_dp = insert_round_robin_sharding(graph, single_br_dp)
        graph, cir_br_dp = insert_round_robin_sharding(graph, cir_br_dp)
        self.assertEqual(find_lca_round_robin_sharding_dp(graph), end_dp)

    def test_non_dispatching_branches(self):
        r"""
        There should be a single DataPipe as the lowest common ancestor of all
        non-dispatching DataPipes that is replaced by ``DummyIterDataPipe``.
        """
        single_br_dp, *_, fork_zip_dp, _, cir_map_dp, _, graph = self._make_dp()
        graph = replace_by_dummy(graph, single_br_dp)
        dps = find_non_dispatching_branches(graph)
        self.assertEqual(len(dps), 2)
        self.assertTrue(all(dp in (fork_zip_dp, cir_map_dp) for dp in dps))

        single_br_dp, multi_br_dp, *_, cir_map_dp, _, graph = self._make_dp()
        graph = replace_by_dummy(graph, multi_br_dp)
        dps = find_non_dispatching_branches(graph)
        self.assertEqual(len(dps), 2)
        self.assertTrue(all(dp in (single_br_dp, cir_map_dp) for dp in dps))

        # In theory, this case should never happen because LCA (fork_zip_dp) should be
        # replaced by _DummpyIterDataPipe if any of child is non-replicable
        single_br_dp, _, _, ch1, ch2, *_, cir_map_dp, _, graph = self._make_dp()
        graph = replace_by_dummy(graph, ch1)
        dps = find_non_dispatching_branches(graph)
        self.assertEqual(len(dps), 3)
        self.assertTrue(all(dp in (single_br_dp, ch2, cir_map_dp) for dp in dps))

        single_br_dp, *_, fork_zip_dp, _, cir_map_dp, _, graph = self._make_dp()
        graph = replace_by_dummy(graph, cir_map_dp)
        dps = find_non_dispatching_branches(graph)
        self.assertTrue(all(dp in (single_br_dp, fork_zip_dp) for dp in dps))

        *_, end_dp, graph = self._make_dp()
        graph = replace_by_dummy(graph, end_dp)
        dps = find_non_dispatching_branches(graph)
        self.assertEqual(len(dps), 0)

        single_br_dp, *_, fork_zip_dp, _, cir_map_dp, _, graph = self._make_dp()
        graph = replace_by_dummy(graph, fork_zip_dp)
        dps = find_non_dispatching_branches(graph)
        self.assertEqual(len(dps), 2)
        self.assertTrue(all(dp in (single_br_dp, cir_map_dp) for dp in dps))

    def test_single_non_replicable_dp(self):
        # All replicable
        *_, end_dp, graph = self._make_dp()
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 1)
        self.assertEqual(dps[0], end_dp)

        # Test the production use case where the last DataPipe is fullsync
        *_, end_dp, _ = self._make_dp()
        dp = end_dp.fullsync()
        graph = traverse_dps(dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 1)
        self.assertEqual(dps[0], end_dp)

        single_br_dp, *_, fork_zip_dp, _, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(single_br_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 2)
        self.assertTrue(all(dp in (fork_zip_dp, cir_map_dp) for dp in dps))

        single_br_dp, *_, ch1, ch2, fork_zip_dp, _, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(fork_zip_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 4)
        self.assertTrue(all(dp in (single_br_dp, ch1, ch2, cir_map_dp) for dp in dps))

        single_br_dp, _, forker_dp, ch1, *_, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(ch1)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 3)
        self.assertTrue(all(dp in (single_br_dp, forker_dp, cir_map_dp) for dp in dps))

        single_br_dp, *_, fork_zip_dp, cir_br_dp, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(cir_map_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 3)
        self.assertTrue(all(dp in (single_br_dp, fork_zip_dp, cir_br_dp) for dp in dps))

        single_br_dp, *_, fork_zip_dp, _, cir_map_dp, end_dp, graph = self._make_dp()
        make_non_replicable_dp(end_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 3)
        self.assertTrue(all(dp in (single_br_dp, fork_zip_dp, cir_map_dp) for dp in dps))

    def test_multi_non_replicable_dps(self):
        single_br_dp, multi_br_dp, *_, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(single_br_dp)
        make_non_replicable_dp(multi_br_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 1)
        self.assertEqual(dps[0], cir_map_dp)

        single_br_dp, _, forker_dp, ch1, *_, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(single_br_dp)
        make_non_replicable_dp(ch1)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 2)
        self.assertTrue(all(dp in (forker_dp, cir_map_dp) for dp in dps))

        single_br_dp, *_, ch1, ch2, fork_zip_dp, _, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(single_br_dp)
        make_non_replicable_dp(fork_zip_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 3)
        self.assertTrue(all(dp in (ch1, ch2, cir_map_dp) for dp in dps))

        single_br_dp, *_, fork_zip_dp, cir_br_dp, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(single_br_dp)
        make_non_replicable_dp(cir_map_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 2)
        self.assertTrue(all(dp in (fork_zip_dp, cir_br_dp) for dp in dps))

        single_br_dp, multi_br_dp, forker_dp, ch1, *_, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(forker_dp)
        make_non_replicable_dp(ch1)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 3)
        self.assertTrue(all(dp in (single_br_dp, multi_br_dp, cir_map_dp) for dp in dps))

        single_br_dp, multi_br_dp, forker_dp, *_, cir_br_dp, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(forker_dp)
        make_non_replicable_dp(cir_map_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 3)
        self.assertTrue(all(dp in (single_br_dp, multi_br_dp, cir_br_dp) for dp in dps))

        single_br_dp, *_, ch1, ch2, fork_zip_dp, cir_br_dp, cir_map_dp, _, graph = self._make_dp()
        make_non_replicable_dp(fork_zip_dp)
        make_non_replicable_dp(cir_map_dp)
        dps = _find_replicable_branches(graph)
        self.assertEqual(len(dps), 4)
        self.assertTrue(all(dp in (single_br_dp, ch1, ch2, cir_br_dp) for dp in dps))


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
