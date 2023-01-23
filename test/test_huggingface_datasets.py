# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import expecttest

from torchdata.datapipes.iter import HuggingFaceHubReader

try:
    import datasets

    HAS_DATASETS = True

except ImportError:
    HAS_DATASETS = False
skipIfNoDatasets = unittest.skipIf(not HAS_DATASETS, "no datasets")


class TestHuggingFaceHubReader(expecttest.TestCase):
    @skipIfNoDatasets
    @patch("datasets.load_dataset")
    def test_huggingface_hubreader(self, mock_load_dataset):
        mock_load_dataset.return_value = datasets.Dataset.from_dict(
            {
                "id": ["7bd227d9-afc9-11e6-aba1-c4b301cdf627", "7bd22905-afc9-11e6-a5dc-c4b301cdf627"],
                "package_name": ["com.mantz_it.rfanalyzer"] * 2,
            }
        )

        datapipe = HuggingFaceHubReader("lhoestq/demo1", revision="branch", streaming=False, use_auth_token=True)

        iterator = iter(datapipe)
        elem = next(iterator)
        assert type(elem) is dict
        assert elem["id"] == "7bd227d9-afc9-11e6-aba1-c4b301cdf627"
        assert elem["package_name"] == "com.mantz_it.rfanalyzer"
        mock_load_dataset.assert_called_with(
            path="lhoestq/demo1", streaming=False, split="train", revision="branch", use_auth_token=True
        )
        with self.assertRaises(StopIteration):
            next(iterator)
            next(iterator)


if __name__ == "__main__":
    unittest.main()
