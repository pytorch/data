#!/bin/bash

# CentOS8 comes with old python (3.6), let's install the new one

set -xe # better be safe than sorry

# As of Aug 2021, CentOS8 don't come with these headers that CPython depends on
sudo dnf install libffi-devel
sudo dnf install bzip2-devel

PY_VERSION=${PY_VERSION:-3.8.8}
wget $(fwdproxy-config wget) https://www.python.org/ftp/python/${PY_VERSION}/Python-${PY_VERSION}.tgz
tar -xf Python-${PY_VERSION}.tgz
cd Python-${PY_VERSION}
./configure --enable-optimizations
make -j 20
sudo make altinstall
