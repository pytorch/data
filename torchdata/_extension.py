# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.machinery
import os
from pathlib import Path


_LIB_DIR = Path(__file__).parent


def _init_extension():
    lib_dir = os.path.dirname(__file__)

    # TODO(631): If any extension had dependency of shared library,
    #       in order to support load these shred libraries dynamically,
    #       we need to add logic to load dll path on Windows
    #       See: https://github.com/pytorch/pytorch/blob/master/torch/__init__.py#L56-L140

    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)  # type: ignore[arg-type]
    ext_specs = extfinder.find_spec("_torchdata")

    if ext_specs is None:
        return

    from torchdata import _torchdata as _torchdata


_init_extension()
