# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import distutils.sysconfig
import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools.command.build_ext import build_ext


__all__ = [
    "get_ext_modules",
    "CMakeBuild",
]


_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()


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
_USE_SYSTEM_AWS_SDK_CPP = _get_build("USE_SYSTEM_AWS_SDK_CPP", False)
_USE_SYSTEM_PYBIND11 = _get_build("USE_SYSTEM_PYBIND11", False)
_USE_SYSTEM_LIBS = _get_build("USE_SYSTEM_LIBS", False)


try:
    # Use the pybind11 from third_party
    if not (_USE_SYSTEM_PYBIND11 or _USE_SYSTEM_LIBS):
        sys.path.insert(0, str(_ROOT_DIR / "third_party/pybind11/"))
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    from setuptools import Extension as Pybind11Extension


def get_ext_modules():
    if _BUILD_S3:
        return [Pybind11Extension(name="torchdata._torchdata", sources=[])]
    else:
        return []


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        # Because the following `cmake` command will build all of `ext_modules`` at the same time,
        # we would like to prevent multiple calls to `cmake`.
        # Therefore, we call `cmake` only for `torchdata._torchdata`,
        # in case `ext_modules` contains more than one module.
        if ext.name != "torchdata._torchdata":
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",  # For Windows
            f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
            f"-DBUILD_S3:BOOL={'ON' if _BUILD_S3 else 'OFF'}",
            f"-DUSE_SYSTEM_AWS_SDK_CPP:BOOL={'ON' if _USE_SYSTEM_AWS_SDK_CPP else 'OFF'}",
            f"-DUSE_SYSTEM_PYBIND11:BOOL={'ON' if _USE_SYSTEM_PYBIND11 else 'OFF'}",
            f"-DUSE_SYSTEM_LIBS:BOOL={'ON' if _USE_SYSTEM_LIBS else 'OFF'}",
        ]

        build_args = ["--config", cfg]

        # Default to Ninja
        if "CMAKE_GENERATOR" not in os.environ or platform.system() == "Windows":
            cmake_args += ["-GNinja"]
        if platform.system() == "Windows":
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
                build_args += [f"-j{self.parallel}"]

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
