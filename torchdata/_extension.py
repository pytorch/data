import ctypes
import importlib
import os
import sys
from pathlib import Path


_LIB_DIR = Path(__file__).parent


def _init_extension():
    if sys.platform == "win32":
        lib_dir = os.path.dirname(__file__)
        py_dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
        aws_dll_path = os.path.join(_LIB_DIR.parent.resolve(), "aws-sdk-cpp", "sdk-lib", "lib")
        # When users create a virtualenv that inherits the base environment,
        # we will need to add the corresponding library directory into
        # DLL search directories. Otherwise, it will rely on `PATH` which
        # is dependent on user settings.
        if sys.exec_prefix != sys.base_exec_prefix:
            base_py_dll_path = os.path.join(sys.base_exec_prefix, "Library", "bin")
        else:
            base_py_dll_path = ""

        dll_paths = list(filter(os.path.exists, [lib_dir, aws_dll_path, py_dll_path, base_py_dll_path]))

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p

        for dll_path in dll_paths:
            if sys.version_info >= (3, 8):
                print("Add all dll directory", dll_path)
                os.add_dll_directory(dll_path)
            elif with_load_library_flags:
                print("Kernel 32 add dll directory", dll_path)
                res = kernel32.AddDllDirectory(dll_path)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
                    raise err

        import glob

        dlls = glob.glob(os.path.join(aws_dll_path, "*.dll"))
        path_patched = False
        for dll in dlls:
            print("DLL", dll)
            is_loaded = False
            if with_load_library_flags:
                res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
                last_error = ctypes.get_last_error()
                if res is None and last_error != 126:
                    err = ctypes.WinError(last_error)
                    err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                    raise err
                elif res is not None:
                    is_loaded = True
            if not is_loaded:
                if not path_patched:
                    os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                    path_patched = True
                res = kernel32.LoadLibraryW(dll)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                    raise err

        kernel32.SetErrorMode(prev_error_mode)

    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_torchdata")

    if ext_specs is None:
        return

    from torchdata import _torchdata as _torchdata


_init_extension()
