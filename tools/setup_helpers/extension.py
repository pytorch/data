import distutils.sysconfig
import os
import platform
import subprocess
from pathlib import Path

import torch
from setuptools import Extension
from setuptools.command.build_ext import build_ext

__all__ = [
    "get_ext_modules",
    "CMakeBuild",
]

_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()
_TORCHDATA_DIR = _ROOT_DIR / "torchdata"


def _get_build(var, default=False):
    if var not in os.environ:
        return default

    val = os.environ.get(var, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        print(f"WARNING: Unexpected environment variable value `{var}={val}`. " f"Expected one of {trues + falses}")
    return False


_BUILD_S3 = _get_build("BUILD_S3", False)


def get_ext_modules():
    modules = [
        Extension(name="torchdata.lib.libtorchdata", sources=[]),
        Extension(name="torchdata._torchdata", sources=[]),
    ]
    return modules


# Based off of pybiind cmake_example
# https://github.com/pybind/cmake_example/blob/2440893c60ed2578fb127dc16e6b348fa0be92c1/setup.py
# and torchaudio CMakeBuild()
# https://github.com/pytorch/audio/blob/ece03edc3fc28a1ce2c28ef438d2898ed0a78d3f/tools/setup_helpers/extension.py#L65
class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        # Since two library files (libtorchdata and _torchdata) need to be
        # recognized by setuptools, we instantiate `Extension` twice. (see `get_ext_modules`)
        # This leads to the situation where this `build_extension` method is called twice.
        # However, the following `cmake` command will build all of them at the same time,
        # so, we do not need to perform `cmake` twice.
        # Therefore we call `cmake` only for `torchdata._torchdata`.
        if ext.name != "torchdata._torchdata":
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
            f"-DBUILD_S3:BOOL={'ON' if _BUILD_S3 else 'OFF'}",
        ]
        build_args = ["--target", "install"]

        # Default to Ninja
        if "CMAKE_GENERATOR" not in os.environ or platform.system() == "Windows":
            cmake_args += ["-GNinja"]
        if platform.system() == "Windows":
            import sys

            python_version = sys.version_info
            cmake_args += [
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                f"-DPYTHON_VERSION={python_version.major}.{python_version.minor}",
            ]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split(".")
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = ".".join(without_abi)
        return ext_filename

    # def build_extension(self, ext):
    #     # Since two library files (libtorchdata and _torchdata) need to be
    #     # recognized by setuptools, we instantiate `Extension` twice. (see `get_ext_modules`)
    #     # This leads to the situation where this `build_extension` method is called twice.
    #     # However, the following `cmake` command will build all of them at the same time,
    #     # so, we do not need to perform `cmake` twice.
    #     # Therefore we call `cmake` only for `torchdata._torchdata`.
    #     if ext.name != "torchdata._torchdata":
    #         return

    #     extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

    #     # required for auto-detection of auxiliary "native" libs
    #     if not extdir.endswith(os.path.sep):
    #         extdir += os.path.sep

    #     cfg = "Debug" if self.debug else "Release"

    #     sdk_dir = "C:\\Program Files (x86)\\aws-cpp-sdk-all"

    #     cmake_args = [
    #         f"-DCMAKE_BUILD_TYPE={cfg}",
    #         f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
    #         f"-DCMAKE_INSTALL_PREFIX={extdir}",
    #         "-DCMAKE_VERBOSE_MAKEFILE=ON",
    #         f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
    #         f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
    #         "-DBUILD_TORCHDATA_PYTHON_EXTENSION:BOOL=ON",
    #         # f"-DPYTHON_EXECUTABLE={sys.executable}",
    #         # "-DCMAKE_CXX_FLAGS=-fPIC",
    #         f"-DBUILD_S3:BOOL={'ON' if _BUILD_S3 else 'OFF'}",
    #     ]
    #     build_args = ["--target", "install"]

    #     # build_args = ["--config", cfg]

    #     # Default to Ninja
    #     if "CMAKE_GENERATOR" not in os.environ or platform.system() == "Windows":
    #         cmake_args += ["-GNinja"]
    #     if platform.system() == "Windows":
    #         import sys

    #         python_version = sys.version_info
    #         cmake_args += [
    #             "-DCMAKE_C_COMPILER=cl",
    #             "-DCMAKE_CXX_COMPILER=cl",
    #             f"-DPYTHON_VERSION={python_version.major}.{python_version.minor}",
    #         ]

    #     # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
    #     # across all generators.
    #     if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
    #         # self.parallel is a Python 3 only way to set parallel jobs by hand
    #         # using -j in the build_ext call, not supported by pip or PyPA-build.
    #         if hasattr(self, "parallel") and self.parallel:
    #             # CMake 3.12+ only.
    #             build_args += ["-j{}".format(self.parallel)]

    #     if not os.path.exists(self.build_temp):
    #         os.makedirs(self.build_temp)

    #     subprocess.check_call(["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp)
    #     subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

    # def get_ext_filename(self, fullname):
    #     ext_filename = super().get_ext_filename(fullname)
    #     ext_filename_parts = ext_filename.split(".")
    #     without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
    #     ext_filename = ".".join(without_abi)
    #     return ext_filename
