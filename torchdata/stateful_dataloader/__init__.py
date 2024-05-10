# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# from .stateful import Stateful
# from .stateful_dataloader import StatefulDataLoader

__all__ = ["Stateful", "StatefulDataLoader"]


import importlib


# Lazy import all modules
def __getattr__(name):
    if name == "StatefulDataLoader":
        from .stateful_dataloader import StatefulDataLoader

        return StatefulDataLoader
        # return importlib.import_module(".stateful_dataloader." + name, __name__)
    elif name == "Stateful":
        from .stateful import Stateful

        return Stateful
        # return importlib.import_module(".stateful." + name, __name__)
    else:
        try:
            return importlib.import_module("." + name, __name__)
        except ModuleNotFoundError:
            if name in globals():
                return globals()[name]
            else:
                raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
