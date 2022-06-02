# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import warnings

import expecttest

from _utils._common_utils_for_test import create_temp_dir, create_temp_files, reset_after_n_next_calls

from torchdata.datapipes.iter import HuggingFaceHubReader, IterableWrapper

try:
    import datasets

    HAS_DATASETS = True

except ImportError:
    HAS_DATASETS = False
skipIfNoDatasets = unittest.skipIf(not HAS_DATASETS, "no datasets")


class TestHuggingFaceHubReader(expecttest.TestCase):
    def setUp(self):
        self.temp_dir = create_temp_dir()
        self.temp_files = create_temp_files(self.temp_dir)
        self.temp_sub_dir = create_temp_dir(self.temp_dir.name)
        self.temp_sub_files = create_temp_files(self.temp_sub_dir, 4, False)

        self.temp_dir_2 = create_temp_dir()
        self.temp_files_2 = create_temp_files(self.temp_dir_2)
        self.temp_sub_dir_2 = create_temp_dir(self.temp_dir_2.name)
        self.temp_sub_files_2 = create_temp_files(self.temp_sub_dir_2, 4, False)

    def tearDown(self):
        try:
            self.temp_sub_dir.cleanup()
            self.temp_dir.cleanup()
            self.temp_sub_dir_2.cleanup()
            self.temp_dir_2.cleanup()
        except Exception as e:
            warnings.warn(f"HuggingFace datasets was not able to cleanup temp dir due to {e}")

    @skipIfNoDatasets
    def test_huggingface_hubreader(self):
        datapipe = HuggingFaceHubReader(dataset="lhoestq/demo1", revision="main", streaming=True)
        elem = next(iter(datapipe))
        assert type(elem) is dict
        assert elem["package_name"] == "com.mantz_it.rfanalyzer"


if __name__ == "__main__":
    unittest.main()
