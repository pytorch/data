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

        dp1 = IterableWrapper(range(6))
        dp2 = dp1.shuffle()
        dp3 = dp2.sharding_filter()
        test_dps = [dp3]
        for dp in test_dps:
            rs = PrototypeMultiProcessingReadingService(num_workers=2, prefetch_worker=0, prefetch_mainloop=1)
            # TODO: This non-deterministically breaks when prefetch_mainloop > 0, figure out why
            #       I might need to check if there are requests when `request_resume` is called...
            #       Perhaps whitelist the allowable and request and wait?
            dl = DataLoader2(dp, reading_service=rs)
            res = []
            for i, x in enumerate(dl):
                res.append(x)
                if i in {2}:  # {0, 5, 10}:
                    # dl.reading_service.reset()
                    print("Pausing...")
                    dl.reading_service._pause()
                    print("Resuming...")
                    dl.reading_service._resume()
                    print("Done with resume", flush=True)
                    print(res)
            print(res)

        # TODO: Add a test case when not calling `_resume` will trigger error when `next` is called

    # def test_reading_service_snapshot(self) -> None:
    #     pass
    #
    # def test_dataloader2_snapshot(self) -> None:
    #     pass


if __name__ == "__main__":
    unittest.main()
