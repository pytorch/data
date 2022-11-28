# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import expecttest
from torch.testing._internal.common_utils import IS_SANDCASTLE
from torchdata.datapipes.iter import IterableWrapper, S3FileLister

skipIfSandcastle = unittest.skipIf(IS_SANDCASTLE, "Skip for internal testing")


@skipIfSandcastle
@patch("torchdata._torchdata")
class TestS3FileListerIterDataPipe(expecttest.TestCase):
    def test_list_files(self, mock_torchdata):
        s3handler_mock = MagicMock()
        mock_torchdata.S3Handler.return_value = s3handler_mock
        s3handler_mock.list_files = MagicMock(
            side_effect=[["s3://bucket-name/folder/a.txt", "s3://bucket-name/folder/b.csv"], []]
        )
        s3_prefixes = IterableWrapper(["s3://bucket-name/folder/"])
        dp_s3_urls = S3FileLister(s3_prefixes)
        assert list(dp_s3_urls) == ["s3://bucket-name/folder/a.txt", "s3://bucket-name/folder/b.csv"]

    def test_list_files_with_filter_mask(self, mock_torchdata):
        s3handler_mock = MagicMock()
        mock_torchdata.S3Handler.return_value = s3handler_mock
        s3handler_mock.list_files = MagicMock(
            side_effect=[["s3://bucket-name/folder/a.txt", "s3://bucket-name/folder/b.csv"], []]
        )
        s3_prefixes = IterableWrapper(["s3://bucket-name/folder/"])
        dp_s3_urls = S3FileLister(s3_prefixes, masks="*.csv")
        assert list(dp_s3_urls) == ["s3://bucket-name/folder/b.csv"]
