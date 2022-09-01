# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata import _extension  # noqa: F401

from . import datapipes

janitor = datapipes.utils.janitor

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

__all__ = [
    "datapipes",
    "janitor",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
