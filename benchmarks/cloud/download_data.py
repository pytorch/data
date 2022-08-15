# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

import requests


# TODO: For certain datasets, it should be possible to directly use TorchVision prototype dataset
#       Download once and re-use


def download_dataset(dataset: str, target_dir: str) -> None:

    # TODO: Get URL and filename based on name

    # Example
    url = "https://data.caltech.edu/tindfiles/serve/e41f5188-0b32-41fa-801b-d1e840915e80/"
    file_name = "caltech-101.zip"
    target = os.path.join(target_dir, file_name)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target, "wb") as f:
            f.write(response.raw.read())
