# Copyright (c) Facebook, Inc. and its affiliates.
import io
import expecttest
import os
import unittest
import warnings

from torchdata.datapipes.iter import (
    FileLoader,
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

    def test_on_disk_cache_holder_iterdatapipe(self):
        import hashlib

        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        expected_file_name = os.path.join(self.temp_dir.name, "OnDisk_LICENSE")
        expected_MD5_hash = "4aabe940637d4389eca42ac1a0e874ec"

        file_dp = IterableWrapper([file_url])

        def _filepath_fn(url):
            filename = "OnDisk_" + os.path.basename(url)
            return os.path.join(self.temp_dir.name, filename)

        def _hash_fn(filepath):
            hash_fn = hashlib.md5()
            with open(filepath, "rb") as f:
                chunk = f.read(1024 ** 2)
                while chunk:
                    hash_fn.update(chunk)
                    chunk = f.read(1024 ** 2)
            return hash_fn.hexdigest() == expected_MD5_hash

        cache_dp = file_dp.on_disk_cache(filepath_fn=_filepath_fn, extra_check_fn=_hash_fn, mode="wt")
        cache_dp = cache_dp.open_url().map(fn=lambda x: b''.join(x).decode(), input_col=1)
        cache_dp = cache_dp.end_caching()

        # File doesn't exist on disk
        self.assertFalse(os.path.exists(expected_file_name))
        it = iter(cache_dp)
        path = next(it)
        self.assertTrue(os.path.exists(expected_file_name))

        # File is cached to disk
        self.assertEqual(expected_file_name, path)

        # Validate file without Error
        fl_dp = FileLoader(cache_dp, mode="rb")
        check_hash_dp = fl_dp.check_hash({expected_file_name: expected_MD5_hash}, "md5", rewind=False)
        _ = list(check_hash_dp)


if __name__ == "__main__":
    unittest.main()
