# Copyright (c) Facebook, Inc. and its affiliates.
from torchdata import _extension  # noqa: F401
from . import datapipes

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

__all__ = ["datapipes"]
