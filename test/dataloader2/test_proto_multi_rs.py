# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import unittest
from unittest import TestCase

from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper


class TestPrototypeMultiProcessingReadingService(TestCase):
    r"""
    This tests specific functionalities of PrototypeMultiProcessingReadingService, notably
    `pause`, `resume`, `snapshot`.
    """

    def test_reading_service_pause_resume(self) -> None:
        # TODO: Pause and resume many times to see if I lose any information
        #       Make sure to test FullSync and Prefetcher as well
        #       Use DL2 with Prototype

        # TODO: Check compatibility with prefetch main_loop
        n_elements = 6  # 6 doesn't work when buffer is large

        dp1 = IterableWrapper(range(n_elements))
        dp2 = dp1.shuffle()
        dp3 = dp2.sharding_filter()
        double_pause_dp = dp3.prefetch().prefetch()
        test_dps = [dp3]  # , double_pause_dp]

        for dp in test_dps:

            rs0 = PrototypeMultiProcessingReadingService(num_workers=0, prefetch_worker=0, prefetch_mainloop=1)
            # TODO: Catch warning here about using more than 0 worker

            rs1 = PrototypeMultiProcessingReadingService(num_workers=1, prefetch_worker=0, prefetch_mainloop=0)  # Pass
            rs2 = PrototypeMultiProcessingReadingService(num_workers=1, prefetch_worker=0, prefetch_mainloop=2)  # Hangs
            rs3 = PrototypeMultiProcessingReadingService(num_workers=2, prefetch_worker=0, prefetch_mainloop=0)  # Pass
            rs4 = PrototypeMultiProcessingReadingService(
                num_workers=2, prefetch_worker=2, prefetch_mainloop=0
            )  # prefetch_worker hangs
            rs5 = PrototypeMultiProcessingReadingService(num_workers=2, prefetch_worker=0, prefetch_mainloop=2)
            rs6 = PrototypeMultiProcessingReadingService(
                num_workers=2, prefetch_worker=2, prefetch_mainloop=2
            )  # this fails

            # TODO: There is a bug when prefetcher has fetched everything from source before pause/resume,
            #       But nonetheless resume set `run_prefetcher` back to True, making it hang

            # test_rss = [rs1, rs2, rs3, rs4, rs5, rs6]
            test_rss = [rs2]
            for rs in test_rss:
                dl = DataLoader2(dp, reading_service=rs)
                res = []
                for i, x in enumerate(dl):
                    res.append(x)
                    if i in {2}:  # {2, n_elements - 3}:
                        dl.reading_service._pause()
                        dl.reading_service._resume()
                print(res)

                self.assertEqual(list(range(n_elements)), sorted(res))

        # TODO: Add a test case when not calling `_resume` will trigger error when `next` is called

    # def test_reading_service_snapshot(self) -> None:
    #     pass
    #
    # def test_dataloader2_snapshot(self) -> None:
    #     pass


if __name__ == "__main__":
    unittest.main()
