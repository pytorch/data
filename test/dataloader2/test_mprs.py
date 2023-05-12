# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import unittest
from unittest import TestCase

from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, subtest

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from torchdata.dataloader2 import DataLoader2, DataLoader2Iterator, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe


def _add_one(x: int) -> int:
    return x + 1


# Test DataPipes
n_elements = 10
dp1 = IterableWrapper(range(n_elements)).shuffle().sharding_filter()
double_pause_dp = dp1.prefetch().prefetch()
test_dps = [dp1, double_pause_dp]


mp_ctx_parametrize = parametrize("ctx", mp.get_all_start_methods())
dp_parametrize = parametrize("dp", test_dps)


def _non_dispatching_dp(n_elements=1000):
    dp = IterableWrapper(list(range(n_elements))).shuffle()
    dp = dp.sharding_filter()
    dp = dp.map(_add_one).batch(8)
    return dp


def _dispatching_dp(n_elements=1000):
    dp = IterableWrapper(list(range(n_elements))).shuffle()
    dp = dp.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
    dp = dp.map(_add_one).batch(16)
    return dp


class NonShardableDataPipe(IterDataPipe):
    def __init__(self, dp: IterDataPipe):
        self.dp = dp

    def is_replicable(self):
        return False

    def __iter__(self):
        yield from self.dp


class TestMultiProcessingReadingService(TestCase):
    r"""
    This tests specific functionalities of MultiProcessingReadingService, notably
    `pause`, `resume`, `snapshot`.
    """

    @mp_ctx_parametrize
    @parametrize("dp_fn", [subtest(_non_dispatching_dp, "non_dispatch"), subtest(_dispatching_dp, "dispatch")])
    @parametrize("main_prefetch", [0, 10])
    @parametrize("worker_prefetch", [0, 10])
    def test_early_exit(self, ctx, dp_fn, main_prefetch, worker_prefetch) -> None:
        dp = dp_fn(1000)
        rs = MultiProcessingReadingService(
            num_workers=2,
            main_prefetch_cnt=main_prefetch,
            worker_prefetch_cnt=worker_prefetch,
            multiprocessing_context=ctx,
        )
        dl: DataLoader2 = DataLoader2(dp, reading_service=rs)
        it = iter(dl)
        for _ in range(10):
            _ = next(it)
        dl.shutdown()

    @mp_ctx_parametrize
    @parametrize("dp_fn", [subtest(_non_dispatching_dp, "non_dispatch"), subtest(_dispatching_dp, "dispatch")])
    @parametrize("main_prefetch", [0, 10])
    @parametrize("worker_prefetch", [0, 10])
    def test_exit(self, ctx, dp_fn, main_prefetch, worker_prefetch) -> None:
        dp = dp_fn(1000)
        rs = MultiProcessingReadingService(
            num_workers=2,
            main_prefetch_cnt=main_prefetch,
            worker_prefetch_cnt=worker_prefetch,
            multiprocessing_context=ctx,
        )
        dl: DataLoader2 = DataLoader2(dp, reading_service=rs)
        _ = list(dl)
        dl.shutdown()

    @mp_ctx_parametrize
    def test_reading_service_pause_resume_0_worker(self, ctx) -> None:

        # Functional Test: Verifies that this ReadingService will raise error when `pause/resume` is used
        #                  with `num_workers = 0`
        rs0 = MultiProcessingReadingService(
            num_workers=0, worker_prefetch_cnt=0, main_prefetch_cnt=0, multiprocessing_context=ctx
        )
        dl0: DataLoader2 = DataLoader2(dp1, reading_service=rs0)
        res0 = []
        for i, x in enumerate(dl0):
            res0.append(x)
            if i in {2}:
                with self.assertRaisesRegex(RuntimeError, r"pause"):
                    dl0._pause()
                with self.assertRaisesRegex(RuntimeError, r"resume"):
                    dl0._resume()
        dl0.shutdown()

    @mp_ctx_parametrize
    @dp_parametrize
    @parametrize(
        "n_workers,worker_prefetch_cnt,main_prefetch_cnt",
        [(1, 0, 0), (1, 0, 2), (2, 0, 0), (2, 2, 0), (2, 0, 2), (2, 2, 2)],
    )
    def test_reading_service_pause_resume(self, ctx, dp, n_workers, worker_prefetch_cnt, main_prefetch_cnt) -> None:

        # Functional Test: Testing various configuration of DataPipe/ReadingService to ensure the pipeline
        #                  properly pauses and resumes
        rs = MultiProcessingReadingService(
            num_workers=n_workers,
            worker_prefetch_cnt=worker_prefetch_cnt,
            main_prefetch_cnt=main_prefetch_cnt,
            multiprocessing_context=ctx,
        )
        dl: DataLoader2 = DataLoader2(dp, reading_service=rs)
        res = []
        for i, x in enumerate(dl):
            res.append(x)
            if i in {2, n_elements - 2}:
                dl._pause()
                dl._resume()

        self.assertEqual(
            list(range(n_elements)),
            sorted(res),
            msg=f"The test is failing with '{ctx}', num_workers = {rs.num_workers}, "
            f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, "
            f"main_prefetch_cnt = {rs.main_prefetch_cnt}",
        )
        dl.shutdown()

    @mp_ctx_parametrize
    @dp_parametrize
    @parametrize("n_workers,worker_prefetch_cnt,main_prefetch_cnt", [(2, 0, 1), (2, 1, 0), (2, 0, 0)])
    def test_reading_service_pause_stop_yield(self, ctx, dp, n_workers, worker_prefetch_cnt, main_prefetch_cnt) -> None:

        # Functional Test: Confirms that `dl` will stop yielding elements after `_pause` is called
        rs = MultiProcessingReadingService(
            num_workers=n_workers,
            worker_prefetch_cnt=worker_prefetch_cnt,
            main_prefetch_cnt=main_prefetch_cnt,
            multiprocessing_context=ctx,
        )
        dl: DataLoader2 = DataLoader2(dp, reading_service=rs)
        res = []
        for i, x in enumerate(dl):
            res.append(x)
            if i in {2}:
                dl._pause()
        self.assertEqual(
            3,
            len(res),
            msg=f"The test is failing with '{ctx}', num_workers = {rs.num_workers}, "
            f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
        )
        dl.shutdown()

    @dp_parametrize
    @parametrize("n_workers,worker_prefetch_cnt,main_prefetch_cnt", [(1, 0, 0), (1, 0, 2), (2, 0, 0), (2, 2, 2)])
    def test_reading_service_limit(self, dp, n_workers, worker_prefetch_cnt, main_prefetch_cnt) -> None:

        rs = MultiProcessingReadingService(
            num_workers=n_workers, worker_prefetch_cnt=worker_prefetch_cnt, main_prefetch_cnt=main_prefetch_cnt
        )

        dl: DataLoader2 = DataLoader2(dp, reading_service=rs)
        res = []
        cumulative_res = []
        n_limit = 3

        it: DataLoader2Iterator = iter(dl)
        it.limit(n_limit)
        for x in it:
            res.append(x)
        # Functional Test: Verify that the number of elements yielded equals to the specified limit
        self.assertEqual(
            n_limit,
            len(res),  # 3
            msg=f"The test is failing with default multiprocessing method, "
            f"num_workers = {rs.num_workers}, "
            f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
        )
        cumulative_res.extend(res)

        # Functional Test: Calling `next` after `limit` will trigger `StopIteration`
        with self.assertRaises(StopIteration):
            next(it)

        # Functional Test: Verify that `limit` persists without the need to set it again
        it.resume()
        res = []
        for x in it:
            res.append(x)
        self.assertEqual(
            n_limit,
            len(res),  # 3
            msg=f"The test is failing with default multiprocessing method, "
            f"num_workers = {rs.num_workers}, "
            f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
        )
        cumulative_res.extend(res)

        # Functional Test: Clear the `limit` and yield the rest of the elements
        it.limit(None)
        it.resume()
        res = []
        for x in it:
            res.append(x)
        self.assertEqual(
            n_elements - 2 * n_limit,
            len(res),  # 4
            msg=f"The test is failing with default multiprocessing method, "
            f"num_workers = {rs.num_workers}, "
            f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
        )

        cumulative_res.extend(res)
        self.assertEqual(list(range(n_elements)), sorted(cumulative_res))

        # Functional Test: Setting `limit` to a different value during after each mini-epoch
        dl2: DataLoader2 = DataLoader2(double_pause_dp, reading_service=rs)
        res = []
        it2: DataLoader2Iterator = iter(dl2)
        it2.limit(3)
        for x in it2:
            res.append(x)

        # Limit can be set before `resume`
        it2.limit(4)
        it2.resume()
        for x in it2:
            res.append(x)
        self.assertEqual(7, len(res))

        # Limit can also be set after `resume`, but before the next `for` loop
        it2.resume()
        it2.limit(2)
        for x in it2:
            res.append(x)
        self.assertEqual(9, len(res))

    def test_initial_epoch_checkpointing(self):
        dp = IterableWrapper(range(20)).shuffle().sharding_filter()
        # Note that the second `shuffle` occurs in the main process, which uses a different RNG from
        # the `shuffle` done in the worker processes
        dp = NonShardableDataPipe(dp).shuffle()  # type: ignore[assignment, arg-type]
        rs = MultiProcessingReadingService(num_workers=2)

        # Functional Test: Saving state before iterator is created
        dl: DataLoader2 = DataLoader2(datapipe=dp, reading_service=rs)
        dl.seed(1)
        initial_state = dl.state_dict()
        it1 = iter(dl)

        restored_dl: DataLoader2 = DataLoader2.from_state(initial_state, rs)  # type: ignore[arg-type]
        restored_dl._restore_checkpoint_beginning_of_epoch()
        self.assertEqual(list(it1), list(restored_dl))

        dl.shutdown()
        restored_dl.shutdown()

        # Functional Test: Saving state after iterator is created
        dl = DataLoader2(datapipe=dp, reading_service=rs)
        dl.seed(1)
        it1 = iter(dl)
        initial_state = dl.state_dict()

        restored_dl = DataLoader2.from_state(initial_state, rs)  # type: ignore[arg-type]
        restored_dl._restore_checkpoint_beginning_of_epoch()
        self.assertEqual(list(it1), list(restored_dl))

        dl.shutdown()
        restored_dl.shutdown()

        # Functional Test: Saving state after iterator is created and began iterating
        dl = DataLoader2(datapipe=dp, reading_service=rs)
        dl.seed(1)
        it1 = iter(dl)
        temp = next(it1)  # Starts iterating
        initial_state = dl.state_dict()

        restored_dl = DataLoader2.from_state(initial_state, rs)  # type: ignore[arg-type]
        restored_dl._restore_checkpoint_beginning_of_epoch()

        self.assertEqual([temp] + list(it1), list(restored_dl))  # Note skipping over 1st element from actual result

        dl.shutdown()
        restored_dl.shutdown()

    # TODO: Test cases when there is official support of `pause` and `resume` with round-robin sharding
    #       Currently, using sharding_round_robin raises a warning
    # def test_round_robin_dispatching_pause_limit(self):
    #     source_dp = IterableWrapper(range(20))
    #     dp = source_dp.shuffle().sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
    #     dp = dp.map(_add_one)

    # TODO: This doesn't work with `num_workers > 1`
    # TODO: Try checking if `dp_list`'s elements are _IterateQueueDP or QueueWrapper, we can safely assume
    #       those DPs belong to a dispatching process and only do pause if worker_id == 0
    #       There might still be a race condition, need to look into the messages

    # rs1 = MultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=0)
    # rs2 = MultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=2)
    # rs3 = MultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=0)
    # rs4 = MultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=2)
    # rss = [rs1, rs2, rs3, rs4]

    # for n, rs in enumerate(rss):
    #     dl = DataLoader2(dp, reading_service=rs)
    #     res = []
    #     # cumulative_res = []
    #     n_limit = 3
    #
    #     it: DataLoader2Iterator = iter(dl)
    #     it.limit(n_limit)  # The `pause` call here doesn't stop
    #     for x in it:
    #         res.append(x)
    #
    #     print()
    #     print(res)
    #
    #     dl.shutdown()

    # # Functional Test: Verify that the number of elements yielded equals to the specified limit
    # # self.assertEqual(
    # #     n_limit,
    # #     len(res),  # 3
    # #     msg=f"The test is failing for rs{n + 1} with default multiprocessing method, "
    # #         f"num_workers = {rs.num_workers}, "
    # #         f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
    # # )
    # cumulative_res.extend(res)
    #
    # # Functional Test: Calling `next` after `limit` will trigger `StopIteration`
    # with self.assertRaisesRegex(StopIteration, "pause"):
    #     next(it)
    #
    # # Functional Test: Verify that `limit` persists without the need to set it again
    # it.resume()
    # res = []
    # for x in it:
    #     res.append(x)
    # # self.assertEqual(
    # #     n_limit,
    # #     len(res),  # 3
    # #     msg=f"The test is failing for rs{n + 1} with default multiprocessing method, "
    # #         f"num_workers = {rs.num_workers}, "
    # #         f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
    # # )
    # cumulative_res.extend(res)
    #
    # # Functional Test: Clear the `limit` and yield the rest of the elements
    # it.limit(None)
    # it.resume()
    # res = []
    # for x in it:
    #     res.append(x)
    # # self.assertEqual(
    # #     n_elements - 2 * n_limit,
    # #     len(res),  # 4
    # #     msg=f"The test is failing for rs{n + 1} with default multiprocessing method, "
    # #         f"num_workers = {rs.num_workers}, "
    # #         f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
    # # )
    #
    # cumulative_res.extend(res)
    # self.assertEqual(list(range(n_elements)), sorted(cumulative_res))

    # TODO: Implemented in an upcoming PR
    # def test_reading_service_snapshot(self) -> None:
    #     pass
    #
    # def test_dataloader2_snapshot(self) -> None:
    #     pass


instantiate_parametrized_tests(TestMultiProcessingReadingService)


if __name__ == "__main__":
    unittest.main()
