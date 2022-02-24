import os
import sys
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
_BUILD_PYTHON_VERSION = os.environ.get("BUILD_PYTHON_VERSION", None)


def get_ext_modules():
    if _BUILD_S3:
        return [Extension(name="torchdata._torchdata", sources=[])]
    else:
        return []


# Based off of pybiind cmake_example
# https://github.com/pybind/cmake_example/blob/2440893c60ed2578fb127dc16e6b348fa0be92c1/setup.py
# and torchaudio CMakeBuild()
# https://github.com/pytorch/audio/blob/ece03edc3fc28a1ce2c28ef438d2898ed0a78d3f/tools/setup_helpers/extension.py#L65
class CMakeBuild(build_ext):
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
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_CXX_FLAGS=-fPIC",
            f"-DBUILD_S3:BOOL={'ON' if _BUILD_S3 else 'OFF'}",
        ]

        build_args = ["--config", cfg]

        if _BUILD_PYTHON_VERSION:
            cmake_args += [
                f"-DBUILD_PYTHON_VERSION={_BUILD_PYTHON_VERSION}",
            ]

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]
        
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
