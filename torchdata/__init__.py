# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

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


_DEPRECATED = ["janitor", "datapipes", "dataloader2"]
_warning_shown = False


# Lazy import all modules
def __getattr__(name):
    if name in _DEPRECATED and not _warning_shown:
        global _warning_shown
        _warning_shown = True

        import warnings

        warnings.warn(
            f"The {_DEPRECATED} modules are deprecated and will be removed in a future "
            f"torchdata release! Please see https://github.com/pytorch/data/issues/1196 "
            f"to learn more and leave feedback.",
            stacklevel=2,
        )
    if name == "janitor":
        return importlib.import_module(".datapipes.utils." + name, __name__)
    else:
        try:
            return importlib.import_module("." + name, __name__)
        except ModuleNotFoundError:
            if name in globals():
                return globals()[name]
            else:
                raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
