import importlib
import os
from pathlib import Path

from torchdata._internal import module_utils as _mod_utils  # noqa: F401

_LIB_DIR = Path(__file__).parent


def _init_extension():
    # load the pybind11 extension
    lib_dir = os.path.dirname(__file__)

    if os.name == "nt":
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
    if ext_specs is not None:
        from torchdata import _torchdata


_init_extension()
