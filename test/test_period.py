# Copyright (c) Facebook, Inc. and its affiliates.
import io
import expecttest
import os
import unittest

from torchdata.datapipes.iter import (
    GDriveReader,
    IterableWrapper,
    OnlineReader,
)


# This TestCase is created due to the limited quota to access google drive
class TestDataPipePeriod(expecttest.TestCase):
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

    def test_online_iterdatapipe(self):

        license_file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        amazon_review_url = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"
        expected_license_file_name = "LICENSE"
        expected_amazon_file_name = "amazon_review_polarity_csv.tar.gz"
        expected_license_MD5_hash = "4aabe940637d4389eca42ac1a0e874ec"
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


if __name__ == "__main__":
    unittest.main()
