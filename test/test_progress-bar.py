# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import expecttest

from torchdata.datapipes.iter import StreamReader, IterableWrapper, HttpReader, Saver
from torchdata.datapipes.iter.util.progress_bar import ProgressBar


class TestIterDataPipeSerialization(expecttest.TestCase):
    def test_progress_bar(self):
        iw = IterableWrapper(
            [
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            ]
        )
        hr = HttpReader(iw)
        sr = StreamReader(hr, chunk=1024 * 1024)
        pb = sr.show_progress(update_fn=lambda data: len(data[1]), reset_fn=lambda data: len(data[1]) < 1024 * 1024)
        dp = Saver(pb, mode="wb", filepath_fn=lambda url: f"./{url.split('/')[-1]}")

        list(dp)


if __name__ == "__main__":
    unittest.main()
