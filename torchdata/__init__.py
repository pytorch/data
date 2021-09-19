# Copyright (c) Facebook, Inc. and its affiliates.
from torchdata import datapipes

import torch

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass