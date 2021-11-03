# Copyright (c) Facebook, Inc. and its affiliates.
import io
import expecttest
import os
import unittest
import warnings

from functools import partial

from torchdata.datapipes.iter import (
    HttpReader,
    IterableWrapper,
)

from _utils._common_utils_for_test import (
    check_hash_fn,
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
        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        prefix = "OnDisk_"
        expected_file_name = os.path.join(self.temp_dir.name, prefix + "LICENSE")
        expected_MD5_hash = "4aabe940637d4389eca42ac1a0e874ec"

        file_dp = IterableWrapper([file_url])

        def _filepath_fn(url):
            filename = prefix + os.path.basename(url)
            return os.path.join(self.temp_dir.name, filename)

        cache_dp = file_dp.on_disk_cache(filepath_fn=_filepath_fn, extra_check_fn=partial(check_hash_fn, expected_hash=expected_MD5_hash, hash_type="md5"))

        # DataPipe Constructor
        cache_dp = HttpReader(cache_dp)
        # Functional API
        cache_dp = cache_dp.map(fn=lambda x: b''.join(x), input_col=1)

        # Start iteration without `end_caching`
        with self.assertRaisesRegex(RuntimeError, "Please call"):
            _ = list(cache_dp)

        cache_dp = cache_dp.end_caching(mode="wb")

        # File doesn't exist on disk
        self.assertFalse(os.path.exists(expected_file_name))
        path = list(cache_dp)[0]
        self.assertTrue(os.path.exists(expected_file_name))

        # File is cached to disk
        self.assertEqual(expected_file_name, path)
        self.assertTrue(check_hash_fn(path, expected_MD5_hash, hash_type="md5"))

        # Call `end_caching` again
        with self.assertRaisesRegex(RuntimeError, "Incomplete `OnDiskCacheHolder` is required in the pipeline"):
            cache_dp = cache_dp.end_caching()

        # TODO(ejguan): Multiple CacheHolders or nested CacheHolders


if __name__ == "__main__":
    unittest.main()
