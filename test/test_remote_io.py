# Copyright (c) Facebook, Inc. and its affiliates.
import io
import expecttest
import os
import unittest
import warnings

from torch.testing._internal.common_utils import slowTest
from torchdata.datapipes.iter import (
    HttpReader,
    IterableWrapper,
)

from _utils._common_utils_for_test import (
    create_temp_dir,
)


class TestDataPipeRemoteIO(expecttest.TestCase):
    def setUp(self):
        self.temp_dir = create_temp_dir()

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(f"TestDataPipeRemoteIO was not able to cleanup temp dir due to {e}")

    def test_http_reader_iterdatapipe(self):

        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        expected_file_name = "LICENSE"
        expected_MD5_hash = "4aabe940637d4389eca42ac1a0e874ec"
        http_reader_dp = HttpReader(IterableWrapper([file_url]))

        # Functional Test: test if the Http Reader can download and read properly
        reader_dp = http_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        self.assertEqual(expected_file_name, os.path.basename(path))
        self.assertTrue(b"BSD" in line)

        # Reset Test: http_reader_dp has been read, but we reset when calling check_hash()
        check_cache_dp = http_reader_dp.check_hash({file_url: expected_MD5_hash}, "md5", rewind=False)
        it = iter(check_cache_dp)
        path, stream = next(it)
        self.assertEqual(expected_file_name, os.path.basename(path))
        self.assertTrue(io.BufferedReader, type(stream))

        # __len__ Test: returns the length of source DataPipe
        source_dp = IterableWrapper([file_url])
        http_dp = HttpReader(source_dp)
        self.assertEqual(1, len(http_dp))


if __name__ == "__main__":
    unittest.main()
