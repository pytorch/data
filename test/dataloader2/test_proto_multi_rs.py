# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest import TestCase

from torch.utils.data.datapipes.iter.grouping import SHARDING_PRIORITIES
from torchdata.dataloader2 import DataLoader2, DataLoader2Iterator, PrototypeMultiProcessingReadingService

from torchdata.datapipes.iter import IterableWrapper


def _add_one(x: int) -> int:
    return x + 1


class TestPrototypeMultiProcessingReadingService(TestCase):
    r"""
    This tests specific functionalities of PrototypeMultiProcessingReadingService, notably
    `pause`, `resume`, `snapshot`.
    """

    # Test DataPipes
    n_elements = 10
    dp1 = IterableWrapper(range(n_elements)).shuffle().sharding_filter()
    double_pause_dp = dp1.prefetch().prefetch()
    test_dps = [dp1, double_pause_dp]
    mp_start_methods = ["fork", "forkserver", "spawn"]

    def test_reading_service_pause_resume_0_worker(self) -> None:

        # Functional Test: Verifies that this ReadingService will raise error when `pause/resume` is used
        #                  with `num_workers = 0`
        for mp_method in self.mp_start_methods:
            rs0 = PrototypeMultiProcessingReadingService(
                num_workers=0, worker_prefetch_cnt=0, main_prefetch_cnt=0, multiprocessing_context=mp_method
            )
            dl0: DataLoader2 = DataLoader2(self.dp1, reading_service=rs0)
            res0 = []
            for i, x in enumerate(dl0):
                res0.append(x)
                if i in {2}:
                    with self.assertRaisesRegex(RuntimeError, r"pause"):
                        dl0._pause()
                    with self.assertRaisesRegex(RuntimeError, r"resume"):
                        dl0._resume()
            dl0.shutdown()

    def test_reading_service_pause_resume(self) -> None:

        for mp_method in self.mp_start_methods:
            rs1 = PrototypeMultiProcessingReadingService(
                num_workers=1, worker_prefetch_cnt=0, main_prefetch_cnt=0, multiprocessing_context=mp_method
            )
            rs2 = PrototypeMultiProcessingReadingService(
                num_workers=1, worker_prefetch_cnt=0, main_prefetch_cnt=2, multiprocessing_context=mp_method
            )
            rs3 = PrototypeMultiProcessingReadingService(
                num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=0, multiprocessing_context=mp_method
            )
            rs4 = PrototypeMultiProcessingReadingService(
                num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=0, multiprocessing_context=mp_method
            )
            rs5 = PrototypeMultiProcessingReadingService(
                num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=2, multiprocessing_context=mp_method
            )
            rs6 = PrototypeMultiProcessingReadingService(
                num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=2, multiprocessing_context=mp_method
            )
            test_rss = [rs1, rs2, rs3, rs4, rs5, rs6]

            for dp in self.test_dps:
                # Functional Test: Testing various configuration of DataPipe/ReadingService to ensure the pipeline
                #                  properly pauses and resumes
                for n, rs in enumerate(test_rss):
                    dl: DataLoader2 = DataLoader2(dp, reading_service=rs)
                    res = []
                    for i, x in enumerate(dl):
                        res.append(x)
                        if i in {2, self.n_elements - 2}:
                            dl._pause()
                            dl._resume()

                    self.assertEqual(
                        list(range(self.n_elements)),
                        sorted(res),
                        msg=f"The test is failing for rs{n + 1} with '{mp_method}', num_workers = {rs.num_workers}, "
                        f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, "
                        f"main_prefetch_cnt = {rs.main_prefetch_cnt}",
                    )
                    dl.shutdown()

    def test_reading_service_pause_stop_yield(self) -> None:

        # Functional Test: Confirms that `dl` will stop yielding elements after `_pause` is called

        for mp_method in self.mp_start_methods:
            rs7 = PrototypeMultiProcessingReadingService(
                num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=1, multiprocessing_context=mp_method
            )
            rs8 = PrototypeMultiProcessingReadingService(
                num_workers=2, worker_prefetch_cnt=1, main_prefetch_cnt=0, multiprocessing_context=mp_method
            )
            rs9 = PrototypeMultiProcessingReadingService(
                num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=0, multiprocessing_context=mp_method
            )

            test_rss2 = [rs7, rs8, rs9]

            for n, rs in enumerate(test_rss2):
                dl: DataLoader2 = DataLoader2(self.double_pause_dp, reading_service=rs)
                res = []
                for i, x in enumerate(dl):
                    res.append(x)
                    if i in {2}:
                        dl._pause()
                self.assertEqual(
                    3,
                    len(res),
                    msg=f"The test is failing for rs{n + 7} with '{mp_method}', num_workers = {rs.num_workers}, "
                    f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
                )
                dl.shutdown()

    def test_reading_service_limit(self) -> None:

        rs1 = PrototypeMultiProcessingReadingService(num_workers=1, worker_prefetch_cnt=0, main_prefetch_cnt=0)
        rs2 = PrototypeMultiProcessingReadingService(num_workers=1, worker_prefetch_cnt=0, main_prefetch_cnt=2)
        rs3 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=0)
        rs4 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=2)
        test_rss = [rs1, rs2, rs3, rs4]

        for dp in self.test_dps:
            for n, rs in enumerate(test_rss):
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
                    msg=f"The test is failing for rs{n + 7} with default multiprocessing method, "
                    f"num_workers = {rs.num_workers}, "
                    f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
                )
                cumulative_res.extend(res)

                # Functional Test: Calling `next` after `limit` will trigger `StopIteration`
                with self.assertRaisesRegex(StopIteration, "pause"):
                    next(it)

                # Functional Test: Verify that `limit` persists without the need to set it again
                it.resume()
                res = []
                for x in it:
                    res.append(x)
                self.assertEqual(
                    n_limit,
                    len(res),  # 3
                    msg=f"The test is failing for rs{n + 7} with default multiprocessing method, "
                    f"num_workers = {rs.num_workers}, "
                    f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
                )
                cumulative_res.extend(res)

                # Functional Test: Clear the `limit` and yield the rest of the elements
                it.clear_limit()
                it.resume()
                res = []
                for x in it:
                    res.append(x)
                self.assertEqual(
                    self.n_elements - 2 * n_limit,
                    len(res),  # 4
                    msg=f"The test is failing for rs{n + 7} with default multiprocessing method, "
                    f"num_workers = {rs.num_workers}, "
                    f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
                )

                cumulative_res.extend(res)
                self.assertEqual(list(range(self.n_elements)), sorted(cumulative_res))

        # Functional Test: Setting `limit` to a different value during after each mini-epoch
        dl2: DataLoader2 = DataLoader2(self.double_pause_dp, reading_service=rs4)
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

<<<<<<< HEAD
    # TODO: Either do more testing or remove mark it as a known issue, telling users to not use `round_robin`
    #       Should clean up first before landing
    def test_dispatching(self):
        source_dp = IterableWrapper(range(20))
        # dp = source_dp.shuffle().sharding_filter()
        dp = source_dp.shuffle().sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
        dp = dp.map(_add_one)

        # TODO: This doesn't seem to work with `num_workers > 1`
        rs1 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=0)
        # rs2 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=2)
        # rs3 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=0)
        # rs4 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=2)
        rss = [rs1]

        for n, rs in enumerate(rss):
            dl = DataLoader2(dp, reading_service=rs)
            res = []
            # cumulative_res = []
            n_limit = 3

            it: DataLoader2Iterator = iter(dl)
            it.limit(n_limit)  # The `pause` call here doesn't stop
            for x in it:
                res.append(x)

            print()
            print(res)

            dl.shutdown()

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
            # it.clear_limit()
            # it.resume()
            # res = []
            # for x in it:
            #     res.append(x)
            # # self.assertEqual(
            # #     self.n_elements - 2 * n_limit,
            # #     len(res),  # 4
            # #     msg=f"The test is failing for rs{n + 1} with default multiprocessing method, "
            # #         f"num_workers = {rs.num_workers}, "
            # #         f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
            # # )
            #
            # cumulative_res.extend(res)
            # self.assertEqual(list(range(self.n_elements)), sorted(cumulative_res))

    def test_dataloader2_snapshot(self) -> None:

        rs1 = PrototypeMultiProcessingReadingService(num_workers=1, worker_prefetch_cnt=0, main_prefetch_cnt=0)
        # rs2 = PrototypeMultiProcessingReadingService(num_workers=1, worker_prefetch_cnt=0, main_prefetch_cnt=2)
        # rs3 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=0)
        # rs4 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=0)
        # rs5 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=2)
        # rs6 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=2, main_prefetch_cnt=2)

        n_samples_before_snapshot = 3

        n_samples_yielded = 0
        initial_seed_rng = None

        test_rss = [rs1]
        for rs in test_rss:
            dl: DataLoader2 = DataLoader2(self.dp1, reading_service=rs)
            res = []
            for i, x in enumerate(dl):
                res.append(x)
                if i in {n_samples_before_snapshot - 1}:
                    n_samples_yielded, initial_seed_rng = dl._get_naive_datapipe_snapshot()
                    break
            dl.shutdown()
            self.assertEqual(n_samples_before_snapshot, len(res))
            self.assertEqual(n_samples_before_snapshot, n_samples_yielded)

            dl_restored: DataLoader2 = DataLoader2(self.dp1, reading_service=rs)
            dl_restored._restore_naive_datapipe_snapshot(n_samples_yielded, initial_seed_rng)
            restored_res = list(dl_restored)
            self.assertEqual(res, restored_res[0 : n_samples_before_snapshot - 1])  # Check if elements are the same
            self.assertEqual(list(range(self.n_elements)), sorted(restored_res))
            dl_restored.shutdown()

            # TODO: Need to figure out the reset situation within `_simple_graph_snapshot_restoration` and ProtoRS


if __name__ == "__main__":
    unittest.main()
