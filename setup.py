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

from tools.setup_helpers.extension import CMakeBuild, get_ext_modules

ROOT_DIR = Path(__file__).parent.resolve()


################################################################################
# Parameters parsed from environment
################################################################################
RUN_BUILD_DEP = True
for _, arg in enumerate(sys.argv):
    if arg in ["clean", "egg_info", "sdist"]:
        RUN_BUILD_DEP = False


def _get_submodule_folders():
    git_modules_path = ROOT_DIR / ".gitmodules"
    if not os.path.exists(git_modules_path):
        return []
    with open(git_modules_path) as f:
        return [
            os.path.join(ROOT_DIR, line.split("=", 1)[1].strip())
            for line in f.readlines()
            if line.strip().startswith("path")
        ]


def _check_submodules():
    def check_for_files(folder, files):
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            print("Could not find any of {} in {}".format(", ".join(files), folder))
            print("Did you run 'git submodule update --init --recursive --jobs 0'?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (os.path.isdir(folder) and len(os.listdir(folder)) == 0)

    if bool(os.getenv("USE_SYSTEM_LIBS", False)):
        return
    folders = _get_submodule_folders()
    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            import time

            print(" --- Trying to initialize submodules")
            start = time.time()
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=ROOT_DIR)
            end = time.time()
            print(f" --- Submodule initialization took {end - start:.2f} sec")
        except Exception:
            print(" --- Submodule initalization failed")
            print("Please run:\n\tgit submodule update --init --recursive --jobs 0")
            sys.exit(1)
    for folder in folders:
        check_for_files(folder, ["CMakeLists.txt", "Makefile", "setup.py", "LICENSE", "LICENSE.md", "LICENSE.txt"])


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
    "portalocker >= 2.0.0",
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

    if RUN_BUILD_DEP:
        from tools.gen_pyi import gen_pyi

        _check_submodules()
        gen_pyi()

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
        ext_modules=get_ext_modules(),
        cmdclass={
            "build_ext": CMakeBuild,
            "clean": clean,
        },
    )
