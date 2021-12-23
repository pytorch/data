# Copyright (c) Facebook, Inc. and its affiliates.
import io
import expecttest
import os
import unittest
import warnings

from torchdata.datapipes.iter import (
    EndOnDiskCacheHolder,
    FileOpener,
    HttpReader,
    IterableWrapper,
    OnDiskCacheHolder,
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
        tar_file_url = "https://raw.githubusercontent.com/pytorch/data/main/test/_fakedata/csv.tar.gz"
        expected_file_name = os.path.join(self.temp_dir.name, "csv.tar.gz")
        expected_MD5_hash = "42cd45e588dbcf64c65751fbf0228af9"
        tar_hash_dict = {expected_file_name: expected_MD5_hash}

        tar_file_dp = IterableWrapper([tar_file_url])

        with self.assertRaisesRegex(RuntimeError, "Expected `OnDiskCacheHolder` existing"):
            _ = tar_file_dp.end_caching()

        def _filepath_fn(url):
            filename = os.path.basename(url)
            return os.path.join(self.temp_dir.name, filename)

        tar_cache_dp = tar_file_dp.on_disk_cache(
            filepath_fn=_filepath_fn,
            hash_dict=tar_hash_dict,
            hash_type="md5",
        )

        # DataPipe Constructor
        tar_cache_dp = HttpReader(tar_cache_dp)

        # Start iteration without `end_caching`
        with self.assertRaisesRegex(RuntimeError, "Please call"):
            _ = list(tar_cache_dp)

        # Both filepath_fn and same_filepath_fn are set
        with self.assertRaisesRegex(ValueError, "`filepath_fn` is mutually"):
            _ = tar_cache_dp.end_caching(mode="wb", filepath_fn=_filepath_fn, same_filepath_fn=True)

        tar_cache_dp = tar_cache_dp.end_caching(mode="wb", same_filepath_fn=True)

        # File doesn't exist on disk
        self.assertFalse(os.path.exists(expected_file_name))

        path = list(tar_cache_dp)[0]

        # File is cached to disk
        self.assertTrue(os.path.exists(expected_file_name))
        self.assertEqual(expected_file_name, path)
        self.assertTrue(check_hash_fn(expected_file_name, expected_MD5_hash))

        # Modify the downloaded file to trigger downloading again
        with open(expected_file_name, "w") as f:
            f.write("0123456789abcdef")

        self.assertFalse(check_hash_fn(expected_file_name, expected_MD5_hash))
        path = list(tar_cache_dp)[0]
        self.assertTrue(check_hash_fn(expected_file_name, expected_MD5_hash))

        # Call `end_caching` again
        with self.assertRaisesRegex(RuntimeError, "`end_caching` can only be invoked once"):
            _ = tar_cache_dp.end_caching()

        # Multiple filepaths
        def _gen_filepath_fn(tar_path):
            for i in range(3):
                yield os.path.join(os.path.dirname(tar_path), "csv", "{}.csv".format(i))

        # DataPipe Constructor
        file_cache_dp = OnDiskCacheHolder(tar_cache_dp, filepath_fn=_gen_filepath_fn)
        file_cache_dp = FileOpener(file_cache_dp, mode="rb")

        # Functional API
        file_cache_dp = file_cache_dp.read_from_tar()

        def _csv_filepath_fn(csv_path):
            return os.path.join(self.temp_dir.name, "csv", os.path.basename(csv_path))

        # Read and decode
        file_cache_dp = file_cache_dp.map(fn=lambda x: x.read().decode(), input_col=1)

        file_cache_dp = EndOnDiskCacheHolder(file_cache_dp, mode="w", filepath_fn=_csv_filepath_fn, skip_read=True)

        cached_it = iter(file_cache_dp)
        for expected_csv_path in _gen_filepath_fn(expected_file_name):
            # File doesn't exist on disk
            self.assertFalse(os.path.exists(expected_csv_path))

            csv_path = next(cached_it)

            # File is cached to disk
            self.assertTrue(os.path.exists(expected_csv_path))
            self.assertEqual(expected_csv_path, csv_path)


if __name__ == "__main__":
    unittest.main()
