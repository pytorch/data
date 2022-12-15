# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import torchdata
import torchdata.dataloader2
import torchdata.datapipes


def s3_test():
    from torchdata._torchdata import S3Handler


if __name__ == "__main__":
    r"""
    TorchData Smoke Test
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-s3", dest="s3", action="store_false")

    options = parser.parse_args()
    # if options.s3:
    #     s3_test()

    # main()
