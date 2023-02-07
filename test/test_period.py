# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import unittest

import expecttest

from torchdata.datapipes.iter import GDriveReader, IterableWrapper, OnlineReader


# This TestCase is created due to the limited quota to access google drive
class TestDataPipePeriod(expecttest.TestCase):
    def test_gdrive_iterdatapipe(self):

        amazon_review_url = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"
        expected_file_name = "amazon_review_polarity_csv.tar.gz"
        expected_MD5_hash = "fe39f8b653cada45afd5792e0f0e8f9b"
        query_params = {"auth": ("fake_username", "fake_password"), "allow_redirects": True}
        timeout = 120
        gdrive_reader_dp = GDriveReader(IterableWrapper([amazon_review_url]), timeout=timeout, **query_params)

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

        # Error Test: test if the GDrive Reader raises an error when the url is invalid
        error_url = "https://drive.google.com/uc?export=download&id=filedoesnotexist"
        http_error_dp = GDriveReader(IterableWrapper([error_url]), timeout=timeout)
        with self.assertRaisesRegex(
            Exception, r"404.+https://drive.google.com/uc\?export=download&id=filedoesnotexist"
        ):
            next(iter(http_error_dp.readlines()))

        # Feature skip-error Test: test if the GDrive Reader skips urls causing problems
        gdrive_skip_error_dp = GDriveReader(
            IterableWrapper([error_url, amazon_review_url]), timeout=timeout, skip_on_error=True
        )
        reader_dp = gdrive_skip_error_dp.readlines()
        with self.assertWarnsRegex(
            Warning, r"404.+https://drive.google.com/uc\?export=download&id=filedoesnotexist.+skipping"
        ):
            it = iter(reader_dp)
            path, line = next(it)
            self.assertEqual(expected_file_name, os.path.basename(path))
            self.assertTrue(line != b"")

    def test_online_iterdatapipe(self):

        license_file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        amazon_review_url = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"
        expected_license_file_name = "LICENSE"
        expected_amazon_file_name = "amazon_review_polarity_csv.tar.gz"
        expected_license_MD5_hash = "bb9675028dd39d2dd2bf71002b93e66c"
        expected_amazon_MD5_hash = "fe39f8b653cada45afd5792e0f0e8f9b"
        query_params = {"auth": ("fake_username", "fake_password"), "allow_redirects": True}
        timeout = 120

        file_hash_dict = {
            license_file_url: expected_license_MD5_hash,
            expected_amazon_file_name: expected_amazon_MD5_hash,
        }

        # Functional Test: can read from GDrive links
        online_reader_dp = OnlineReader(IterableWrapper([amazon_review_url]), timeout=timeout, **query_params)
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

        # Error Test: test if the Online Reader raises an error when the url is invalid
        error_url_http = "https://github.com/pytorch/data/this/url/dont/exist"
        online_error_dp = OnlineReader(IterableWrapper([error_url_http]), timeout=timeout)
        with self.assertRaisesRegex(Exception, f"404.+{error_url_http}"):
            next(iter(online_error_dp.readlines()))

        error_url_gdrive = "https://drive.google.com/uc?export=download&id=filedoesnotexist"
        online_error_dp = OnlineReader(IterableWrapper([error_url_gdrive]), timeout=timeout)
        with self.assertRaisesRegex(
            Exception, r"404.+https://drive.google.com/uc\?export=download&id=filedoesnotexist"
        ):
            next(iter(online_error_dp.readlines()))

        # Feature skip-error Test: test if the Online Reader skips urls causing problems
        online_skip_error_dp = OnlineReader(
            IterableWrapper([error_url_http, error_url_gdrive, license_file_url]), timeout=timeout, skip_on_error=True
        )
        reader_dp = online_skip_error_dp.readlines()
        with self.assertWarnsRegex(Warning, f"404.+{error_url_http}.+skipping"):
            it = iter(reader_dp)
            path, line = next(it)
            self.assertEqual(expected_license_file_name, os.path.basename(path))
            self.assertTrue(b"BSD" in line)


if __name__ == "__main__":
    unittest.main()
