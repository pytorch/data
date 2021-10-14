# Copyright (c) Facebook, Inc. and its affiliates.
import io
import expecttest
import os

import unittest

from torch.testing._internal.common_utils import slowTest
from torchdata.datapipes.iter import (
    FileLoader,
    GDriveReader,
    HttpReader,
    IterableWrapper,
    OnlineReader,
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
            warnings.warn(f"TestDataPipeRemote was not able to cleanup temp dir due to {e}")

    @slowTest
    def test_http_reader_iterdatapipe(self):

        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        expected_file_name = "LICENSE"
        expected_MD5_hash = "6fc98cce3570de1956f7dbfcb9ca9dd1"
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

    @slowTest
    def test_gdrive_iterdatapipe(self):

        amazon_review_url = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"
        expected_file_name = "amazon_review_polarity_csv.tar.gz"
        expected_MD5_hash = "fe39f8b653cada45afd5792e0f0e8f9b"
        gdrive_reader_dp = GDriveReader(IterableWrapper([amazon_review_url]))

        # Functional Test: test if the GDrive Reader can download and read properly
        reader_dp = gdrive_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        self.assertEqual(expected_file_name, os.path.basename(path))
        self.assertTrue(line != b"")

        # Reset Test: gdrive_reader_dp has been read, but we reset when calling check_hash()
        check_cache_dp = gdrive_reader_dp.check_hash({expected_file_name: expected_MD5_hash}, "md5", rewind=False)
        it = iter(check_cache_dp)
        path, stream = next(it)
        self.assertEqual(expected_file_name, os.path.basename(path))
        self.assertTrue(io.BufferedReader, type(stream))

        # __len__ Test: returns the length of source DataPipe
        source_dp = IterableWrapper([amazon_review_url])
        gdrive_dp = GDriveReader(source_dp)
        self.assertEqual(1, len(gdrive_dp))

    @slowTest
    def test_online_iterdatapipe(self):

        license_file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        amazon_review_url = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"
        expected_license_file_name = "LICENSE"
        expected_amazon_file_name = "amazon_review_polarity_csv.tar.gz"
        expected_license_MD5_hash = "6fc98cce3570de1956f7dbfcb9ca9dd1"
        expected_amazon_MD5_hash = "fe39f8b653cada45afd5792e0f0e8f9b"

        file_hash_dict = {
            license_file_url: expected_license_MD5_hash,
            expected_amazon_file_name: expected_amazon_MD5_hash,
        }

        # Functional Test: can read from GDrive links
        online_reader_dp = OnlineReader(IterableWrapper([amazon_review_url]))
        reader_dp = online_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        self.assertEqual(expected_amazon_file_name, os.path.basename(path))
        self.assertTrue(line != b"")

        # Functional Test: can read from other links
        online_reader_dp = OnlineReader(IterableWrapper([license_file_url]))
        reader_dp = online_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        self.assertEqual(expected_license_file_name, os.path.basename(path))
        self.assertTrue(line != b"")

        # Reset Test: reset online_reader_dp by calling check_hash
        check_cache_dp = online_reader_dp.check_hash(file_hash_dict, "md5", rewind=False)
        it = iter(check_cache_dp)
        path, stream = next(it)
        self.assertEqual(expected_license_file_name, os.path.basename(path))
        self.assertTrue(io.BufferedReader, type(stream))

        # Functional Test: works with multiple URLs of different sources
        online_reader_dp = OnlineReader(IterableWrapper([license_file_url, amazon_review_url]))
        check_cache_dp = online_reader_dp.check_hash(file_hash_dict, "md5", rewind=False)
        it = iter(check_cache_dp)
        for expected_file_name, (path, stream) in zip([expected_license_file_name, expected_amazon_file_name], it):
            self.assertEqual(expected_file_name, os.path.basename(path))
            self.assertTrue(io.BufferedReader, type(stream))

        # __len__ Test: returns the length of source DataPipe
        self.assertEqual(2, len(online_reader_dp))

    def test_on_disk_cache_holder_iterdatapipe(self):
        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        expected_file_name = os.path.join(self.temp_dir.name, "OnDisk_LICENSE")
        expected_MD5_hash = "4aabe940637d4389eca42ac1a0e874ec"

        file_dp = IterableWrapper([file_url])

        def _filepath_fn(url):
            filename = "OnDisk_" + os.path.basename(url)
            return os.path.join(self.temp_dir.name, filename)

        def _cache_check_fn(url):
            import hashlib
            filename = "OnDisk_" + os.path.basename(url)
            filepath = os.path.join(self.temp_dir.name, filename)
            if not os.path.exists(filepath):
                return False

            hash_fn = hashlib.md5()
            with open(filepath, "rb") as f:
                chunk = f.read(1024 ** 2)
                while chunk:
                    hash_fn.update(chunk)
                    chunk = f.read(1024 ** 2)
            return hash_fn.hexdigest() == expected_MD5_hash

        cache_dp = file_dp.on_disk_cache(mode="wt", filepath_fn=_filepath_fn, cache_check_fn=_cache_check_fn).open_url().map(fn=lambda x: b''.join(x).decode(), input_col=1).end_caching()

        self.assertFalse(os.path.exists(expected_file_name))
        it = iter(cache_dp)
        path = next(it)
        self.assertTrue(os.path.exists(expected_file_name))

        self.assertEqual(expected_file_name, path)

        # Validate file without Error
        fl_dp = FileLoader(cache_dp)
        check_hash_dp = fl_dp.check_hash({expected_file_name: expected_MD5_hash}, "md5", rewind=False).map(lambda fd: fd.close(), input_col=1)
        _ = list(check_hash_dp)


if __name__ == "__main__":
    unittest.main()
