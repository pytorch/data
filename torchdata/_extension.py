import os
import warnings
from pathlib import Path

import torch
from torchdata._internal import module_utils as _mod_utils  # noqa: F401

_LIB_DIR = Path(__file__).parent / "lib"


def _get_lib_path(lib: str):
    suffix = "pyd" if os.name == "nt" else "so"
    path = _LIB_DIR / f"{lib}.{suffix}"
    return path


def _load_lib(lib: str):
    path = _get_lib_path(lib)
    # In case `torchdata` is deployed with `pex` format, this file does not exist.
    # In this case, we expect that `libtorchdata` is available somewhere
    # in the search path of dynamic loading mechanism, and importing `_torchdata`,
    # which depends on `libtorchdata` and dynamic loader will handle it for us.
    print("[Extension] _torchdata path: ", path)
    print("[Extension] _torchdata path exists: ", path.exists())
    if path.exists():
        torch.ops.load_library(path)
        torch.classes.load_library(path)


def _init_extension():
    if not _mod_utils.is_module_available("torchdata._torchdata"):
        warnings.warn("torchdata C++ extension is not available.")
        return

    import torchdata
    print("[Extension] torch path: ", torch.__file__)
    print("[Extension] torchdata path: ", torchdata.__file__)

    _load_lib("_torchdata")
    # This import is for initializing the methods registered via PyBind11
    # This has to happen after the base library is loaded
    from torchdata import _torchdata  # noqa


_init_extension()
