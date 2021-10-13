#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import subprocess
from pathlib import Path
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()


def _get_version():
    version = '0.1.0a0'
    sha = 'Unknown'
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT_DIR)).decode('ascii').strip()
    except Exception:
        pass

    os_build_version = os.getenv('BUILD_VERSION')
    if os_build_version:
        version = os_build_version
    elif sha != 'Unknown':
        version += '+' + sha[:7]

    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / 'torchdata' / 'version.py'
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


VERSION, SHA = _get_version()
_export_version(VERSION, SHA)

print('-- Building version ' + VERSION)

pytorch_package_version = os.getenv('PYTORCH_VERSION')

pytorch_package_dep = 'torch'
if pytorch_package_version:
    pytorch_package_dep += "==" + pytorch_package_version

setup(
    # Metadata
    name='torchdata',
    version=VERSION,
    description='Composable data loading modules for PyTorch',
    url='https://github.com/pytorch/data',
    author='PyTorch Team',
    author_email='packages@pytorch.org',
    license='BSD',
    install_requires=['requests', pytorch_package_dep],
    python_requires='>=3.6',
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    # Package Info
    packages=find_packages(exclude=["build*", "test*", "third_party*", "examples"]),
    zip_safe=False,
)
