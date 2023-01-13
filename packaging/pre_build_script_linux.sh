#!/bin/bash
set -ex

source packaging/manylinux/python_helper.sh
yum -y install ninja-build zlib-static
# Docker path is /__w by default
export WORKSPACE="/__w"
# Install static OpenSSL/libcrypto library
./packaging/manylinux/install_openssl_curl.sh

pip install cmake ninja
