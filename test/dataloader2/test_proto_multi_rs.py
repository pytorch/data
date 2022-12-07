# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import multiprocessing as mp
import unittest
from unittest import TestCase

from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService

from torchdata.datapipes.iter import IterableWrapper


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
                        dl0.pause()
                    with self.assertRaisesRegex(RuntimeError, r"resume"):
                        dl0.resume()
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
                            dl.pause()
                            dl.resume()

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
                        dl.pause()
                self.assertEqual(
                    3,
                    len(res),
                    msg=f"The test is failing for rs{n + 7} with '{mp_method}', num_workers = {rs.num_workers}, "
                    f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
                )
                dl.shutdown()

    def test_reading_service_pause_stop_yield(self) -> None:

        # Functional Test: Confirms that `dl` will stop yielding elements after `_pause` is called
        rs7 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=1)
        rs8 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=1, main_prefetch_cnt=0)
        rs9 = PrototypeMultiProcessingReadingService(num_workers=2, worker_prefetch_cnt=0, main_prefetch_cnt=0)

        test_rss2 = [rs7, rs8, rs9]
        for n, rs in enumerate(test_rss2):
            dl: DataLoader2 = DataLoader2(self.double_pause_dp, reading_service=rs)
            res = []
            for i, x in enumerate(dl):
                res.append(x)
                if i in {2}:
                    dl.pause()
            self.assertEqual(
                3,
                len(res),
                msg=f"The test is failing for rs{n + 7}, with num_workers = {rs.num_workers}, "
                f"worker_prefetch_cnt = {rs.worker_prefetch_cnt}, main_prefetch_cnt = {rs.main_prefetch_cnt}",
            )
            dl.shutdown()

    # def test_reading_service_snapshot(self) -> None:
    #     pass

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
