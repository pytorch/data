#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import distutils.command.clean
import os
import shutil
import subprocess
import sys

from pathlib import Path

from setuptools import find_packages, setup

from tools import setup_helpers
from tools.gen_pyi import gen_pyi

ROOT_DIR = Path(__file__).parent.resolve()


def _get_version():
    with open(os.path.join(ROOT_DIR, "version.txt")) as f:
        version = f.readline().strip()

    sha = "Unknown"
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        pass

    os_build_version = os.getenv("BUILD_VERSION")
    if os_build_version:
        version = os_build_version
    elif sha != "Unknown":
        version += "+" + sha[:7]

    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "torchdata" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


# Use new version of torch on main branch
pytorch_package_dep = "torch>1.11.0"
if os.getenv("PYTORCH_VERSION"):
    pytorch_package_dep = pytorch_package_dep.split(">")[0]
    pytorch_package_dep += "==" + os.getenv("PYTORCH_VERSION")


requirements = [
    "urllib3 >= 1.25",
    "requests",
    pytorch_package_dep,
]


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchdata extension
        def remove_extension(pattern):
            for path in (ROOT_DIR / "torchdata").glob(pattern):
                print(f"removing extension '{path}'")
                path.unlink()

        for ext in ["so", "dylib", "pyd"]:
            remove_extension("**/*." + ext)

        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


if __name__ == "__main__":
    VERSION, SHA = _get_version()
    _export_version(VERSION, SHA)

    print("-- Building version " + VERSION)

    if sys.argv[1] != "clean":
        gen_pyi()
        # TODO: Fix #343
        os.chdir(ROOT_DIR)

    setup(
        # Metadata
        name="torchdata",
        version=VERSION,
        description="Composable data loading modules for PyTorch",
        url="https://github.com/pytorch/data",
        author="PyTorch Team",
        author_email="packages@pytorch.org",
        license="BSD",
        install_requires=requirements,
        python_requires=">=3.7",
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        package_data={
            "torchdata": [
                "datapipes/iter/*.pyi",
                "datapipes/map/*.pyi",
            ],
        },
        # Package Info
        packages=find_packages(exclude=["test*", "examples*", "tools*", "torchdata.csrc*", "build*"]),
        zip_safe=False,
        # C++ Extension Modules
        ext_modules=setup_helpers.get_ext_modules(),
        cmdclass={
            "build_ext": setup_helpers.CMakeBuild,
            "clean": clean,
        },
    )
