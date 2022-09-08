# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import time
import unittest

from torch.utils.data.graph import DataPipe

from torchdata.dataloader2 import (
        DataLoader2,
        MultiProcessingReadingService,
        )
from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.map import MapDataPipe


class Test_iter2map_converter(unittest.TestCase):
    @staticmethod
    def _get_multi_reading_service():
        return MultiProcessingReadingService(num_workers=2)

    def test_lazy_load(self) -> None:
        source_dp = IterableWrapper([(i, i) for i in range(10)])
        map_dp = source_dp.to_map_datapipe()
        dl: DataLoader2 = DataLoader2(datapipe=map_dp, reading_service=self._get_multi_reading_service())
        # Lazy loading
        self.assertTrue(dl.datapipe._map is None)
        
        for _ in dl:
            pass
        
        # Lazy loading in multprocessing
        self.assertTrue(dl.datapipe.__dict__['iterable'].dataset.__dict__['_datapipe']._map is None)


if __name__=="__main__":
    unittest.main()
