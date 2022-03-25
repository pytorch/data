import os
import warnings
from pathlib import Path

import torch
from torchdata._internal import module_utils as _mod_utils  # noqa: F401

_LIB_DIR = Path(__file__).parent


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


# def _init_extension():
#     if not _mod_utils.is_module_available("torchdata._torchdata"):
#         warnings.warn("torchdata C++ extension is not available.")
#         return

#     import torchdata
#     print("[Extension] torch path: ", torch.__file__)
#     print("[Extension] torchdata path: ", torchdata.__file__)

#     _load_lib("_torchdata")
#     # This import is for initializing the methods registered via PyBind11
#     # This has to happen after the base library is loaded
#     from torchdata import _torchdata  # noqa


def _init_extension():
    import importlib
    import os

    import torch

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)

    if os.name == "nt":
        # Register the main torchvision library location on the default DLL path
        import ctypes
        import sys

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        if sys.version_info >= (3, 8):
            os.add_dll_directory(lib_dir)
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(lib_dir)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
                raise err

        kernel32.SetErrorMode(prev_error_mode)

    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_torchdata")
    if ext_specs is None:
        raise ImportError("torchdata C++ Extension is not found.")
    torch.ops.load_library(ext_specs.origin)
    torch.classes.load_library(ext_specs.origin)


_init_extension()
