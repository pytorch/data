#!/bin/bash

# CentOS8 comes with old python (3.6), let's install the new one

set -xe # better be safe than sorry

PY_VERSION=${PY_VERSION:-3.8.8}
wget $(fwdproxy-config wget) https://www.python.org/ftp/python/${PY_VERSION}/Python-${PY_VERSION}.tgz
tar -xf Python-${PY_VERSION}.tgz
cd Python-${PY_VERSION}
./configure --enable-optimizations
make -j 20
sudo make altinstall
