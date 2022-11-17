# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import unittest
import warnings

import expecttest

import torchdata

from _utils._common_utils_for_test import check_hash_fn, create_temp_dir, IS_M1, IS_WINDOWS
from torch.utils.data import DataLoader

from torchdata.datapipes.iter import (
    EndOnDiskCacheHolder,
    FileOpener,
    FSSpecFileLister,
    FSSpecFileOpener,
    HttpReader,
    IterableWrapper,
    OnDiskCacheHolder,
    S3FileLister,
    S3FileLoader,
)

try:
    import fsspec

    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False

try:
    import s3fs

    HAS_FSSPEC_S3 = True
except ImportError:
    HAS_FSSPEC_S3 = False
skipIfNoFSSpecS3 = unittest.skipIf(not (HAS_FSSPEC and HAS_FSSPEC_S3), "no FSSpec with S3fs")

try:
    import adlfs

    HAS_FSSPEC_AZ = True
except ImportError:
    HAS_FSSPEC_AZ = False
skipIfNoFSSpecAZ = unittest.skipIf(not (HAS_FSSPEC and HAS_FSSPEC_AZ), "no FSSpec with adlfs")

try:
    from torchdata._torchdata import S3Handler

    HAS_AWS = True
except ImportError:
    HAS_AWS = False
skipIfAWS = unittest.skipIf(HAS_AWS, "AWSSDK Enabled")
skipIfNoAWS = unittest.skipIf(not HAS_AWS, "No AWSSDK Enabled")


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
        expected_MD5_hash = "bb9675028dd39d2dd2bf71002b93e66c"
        query_params = {"auth": ("fake_username", "fake_password"), "allow_redirects": True}
        timeout = 120
        http_reader_dp = HttpReader(IterableWrapper([file_url]), timeout=timeout, **query_params)

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
        self.assertEqual(1, len(http_reader_dp))

        # Error Test: test if the Http Reader raises an error when the url is invalid
        error_url = "https://github.com/pytorch/data/this/url/dont/exist"
        http_error_dp = HttpReader(IterableWrapper([error_url]), timeout=timeout)
        with self.assertRaisesRegex(Exception, "[404]"):
            next(iter(http_error_dp.readlines()))

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

        # Cache decompressed archive but only check root directory
        root_dir = "temp"
        file_cache_dp = OnDiskCacheHolder(
            tar_cache_dp, filepath_fn=lambda tar_path: os.path.join(os.path.dirname(tar_path), root_dir)
        )
        remember_cache_dp_object = file_cache_dp
        file_cache_dp = FileOpener(file_cache_dp, mode="rb").load_from_tar()

        file_cache_dp = file_cache_dp.end_caching(
            mode="wb",
            filepath_fn=lambda file_path: os.path.join(self.temp_dir.name, root_dir, os.path.basename(file_path)),
        )

        cached_it = iter(file_cache_dp)
        for i in range(3):
            expected_csv_path = os.path.join(self.temp_dir.name, root_dir, f"{i}.csv")

            # File doesn't exist on disk
            # Check disabled due to some elements of prefetching inside of on_disck_cache
            # self.assertFalse(os.path.exists(expected_csv_path))

            csv_path = next(cached_it)

            # File is cached to disk
            self.assertTrue(os.path.exists(expected_csv_path))
            self.assertEqual(expected_csv_path, csv_path)

        # This is the situation when previous process had no canche to release promise file on the file lists,
        # as we are in same pid, we need to force iterators to finish by deleting or exhausing them
        del cached_it

        if not IS_WINDOWS:
            dl = DataLoader(file_cache_dp, num_workers=3, multiprocessing_context="fork", batch_size=1)
            expected = [[os.path.join(self.temp_dir.name, root_dir, f"{i}.csv")] for i in range(3)] * 3
            res = list(dl)
            self.assertEqual(sorted(expected), sorted(res))

            remember_cache_dp_object._download_everything = True
            workers = 100
            dl = DataLoader(file_cache_dp, num_workers=workers, multiprocessing_context="fork", batch_size=1)
            expected = [[os.path.join(self.temp_dir.name, root_dir, f"{i}.csv")] for i in range(3)] * workers
            res = list(dl)
            self.assertEqual(sorted(expected), sorted(res))

    @skipIfNoFSSpecS3
    def test_fsspec_io_iterdatapipe(self):
        input_list = [
            (["s3://ai2-public-datasets"], 40),  # bucket without '/'
            (["s3://ai2-public-datasets/charades/"], 18),  # bucket with '/'
            (
                [
                    "s3://ai2-public-datasets/charades/Charades_v1.zip",
                    "s3://ai2-public-datasets/charades/Charades_v1_flow.tar",
                    "s3://ai2-public-datasets/charades/Charades_v1_rgb.tar",
                    "s3://ai2-public-datasets/charades/Charades_v1_480.zip",
                ],
                4,
            ),  # multiple files
        ]
        for urls, num in input_list:
            fsspec_lister_dp = FSSpecFileLister(IterableWrapper(urls), anon=True)
            self.assertEqual(sum(1 for _ in fsspec_lister_dp), num, f"{urls} failed")

        url = "s3://ai2-public-datasets/charades/"
        fsspec_loader_dp = FSSpecFileOpener(FSSpecFileLister(IterableWrapper([url]), anon=True), anon=True)
        res = list(fsspec_loader_dp)
        self.assertEqual(len(res), 18, f"{input} failed")

    @skipIfNoFSSpecAZ
    def test_fsspec_azure_blob(self):
        url = "public/curated/covid-19/ecdc_cases/latest/ecdc_cases.csv"
        account_name = "pandemicdatalake"
        azure_prefixes = ["abfs", "az"]
        fsspec_loader_dp = {}

        for prefix in azure_prefixes:
            fsspec_lister_dp = FSSpecFileLister(f"{prefix}://{url}", account_name=account_name)
            fsspec_loader_dp[prefix] = FSSpecFileOpener(fsspec_lister_dp, account_name=account_name).parse_csv()

        res_abfs = list(fsspec_loader_dp["abfs"])[0]
        res_az = list(fsspec_loader_dp["az"])[0]
        self.assertEqual(res_abfs, res_az, f"{input} failed")

    @skipIfAWS
    def test_disabled_s3_io_iterdatapipe(self):
        file_urls = ["s3://ai2-public-datasets"]
        with self.assertRaisesRegex(ModuleNotFoundError, "TorchData must be built with"):
            _ = S3FileLister(IterableWrapper(file_urls))
        with self.assertRaisesRegex(ModuleNotFoundError, "TorchData must be built with"):
            _ = S3FileLoader(IterableWrapper(file_urls))

    @skipIfNoAWS
    @unittest.skipIf(IS_M1, "PyTorch M1 CI Machine doesn't allow accessing")
    def test_s3_io_iterdatapipe(self):
        # S3FileLister: different inputs
        input_list = [
            [["s3://ai2-public-datasets"], 78],  # bucket without '/'
            [["s3://ai2-public-datasets/"], 78],  # bucket with '/'
            [["s3://ai2-public-datasets/charades"], 18],  # folder without '/'
            [["s3://ai2-public-datasets/charades/"], 18],  # folder without '/'
            [["s3://ai2-public-datasets/charad"], 18],  # prefix
            [
                [
                    "s3://ai2-public-datasets/charades/Charades_v1",
                    "s3://ai2-public-datasets/charades/Charades_vu17",
                ],
                12,
            ],  # prefixes
            [["s3://ai2-public-datasets/charades/Charades_v1.zip"], 1],  # single file
            [
                [
                    "s3://ai2-public-datasets/charades/Charades_v1.zip",
                    "s3://ai2-public-datasets/charades/Charades_v1_flow.tar",
                    "s3://ai2-public-datasets/charades/Charades_v1_rgb.tar",
                    "s3://ai2-public-datasets/charades/Charades_v1_480.zip",
                ],
                4,
            ],  # multiple files
            [
                [
                    "s3://ai2-public-datasets/charades/Charades_v1.zip",
                    "s3://ai2-public-datasets/charades/Charades_v1_flow.tar",
                    "s3://ai2-public-datasets/charades/Charades_v1_rgb.tar",
                    "s3://ai2-public-datasets/charades/Charades_v1_480.zip",
                    "s3://ai2-public-datasets/charades/Charades_vu17",
                ],
                10,
            ],  # files + prefixes
        ]
        for input in input_list:
            s3_lister_dp = S3FileLister(IterableWrapper(input[0]), region="us-west-2")
            self.assertEqual(sum(1 for _ in s3_lister_dp), input[1], f"{input[0]} failed")

        # S3FileLister: prefixes + different region
        file_urls = [
            "s3://aft-vbi-pds/bin-images/111",
            "s3://aft-vbi-pds/bin-images/222",
        ]
        s3_lister_dp = S3FileLister(IterableWrapper(file_urls), request_timeout_ms=10000, region="us-east-1")
        self.assertEqual(sum(1 for _ in s3_lister_dp), 2212, f"{input} failed")

        # S3FileLister: incorrect inputs
        input_list = [
            [""],
            ["ai2-public-datasets"],
            ["s3://"],
            ["s3:///bin-images"],
        ]
        for input in input_list:
            with self.assertRaises(ValueError, msg=f"{input} should raise ValueError."):
                s3_lister_dp = S3FileLister(IterableWrapper(input), region="us-east-1")
                for _ in s3_lister_dp:
                    pass

        # S3FileLoader: loader
        input = [
            "s3://charades-tar-shards/charades-video-0.tar",
            "s3://charades-tar-shards/charades-video-1.tar",
        ]  # multiple files
        s3_loader_dp = S3FileLoader(input, region="us-west-2")
        self.assertEqual(sum(1 for _ in s3_loader_dp), 2, f"{input} failed")

        input = [["s3://aft-vbi-pds/bin-images/100730.jpg"], 1]
        s3_loader_dp = S3FileLoader(input[0], region="us-east-1")
        self.assertEqual(sum(1 for _ in s3_loader_dp), input[1], f"{input[0]} failed")

        # S3FileLoader: incorrect inputs
        input_list = [
            [""],
            ["ai2-public-datasets"],
            ["s3://"],
            ["s3:///bin-images"],
            ["s3://ai2-public-datasets/bin-image"],
        ]
        for input in input_list:
            with self.assertRaises(ValueError, msg=f"{input} should raise ValueError."):
                s3_loader_dp = S3FileLoader(input, region="us-east-1")
                for _ in s3_loader_dp:
                    pass

        # integration test
        input = [["s3://charades-tar-shards/"], 10]
        s3_lister_dp = S3FileLister(IterableWrapper(input[0]), region="us-west-2")
        s3_loader_dp = S3FileLoader(s3_lister_dp, region="us-west-2")
        self.assertEqual(sum(1 for _ in s3_loader_dp), input[1], f"{input[0]} failed")


if __name__ == "__main__":
    unittest.main()
