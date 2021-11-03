# Copyright (c) Meta Platforms, Inc. and its affiliates.
from . import datapipes

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

__all__ = ["datapipes"]
