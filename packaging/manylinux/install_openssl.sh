#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

OPENSSL_URL="https://www.openssl.org/source/"
OPENSSL_NAME="openssl-1.1.1n"
OPENSSL_SHA256="40dceb51a4f6a5275bde0e6bf20ef4b91bfc32ed57c0552e2e8e15463372b17a"
OPENSSL_BUILD_FLAGS="no-ssl2 no-zlib no-shared no-comp no-dynamic-engine enable-ec_nistp_64_gcc_128"

function check_sha256sum {
    local fname=$1
    local sha256=$2
    echo "${sha256}  ${fname}" > ${fname}.sha256
    sha256sum -c ${fname}.sha256
    rm ${fname}.sha256
}

yum erase -y openssl-devel

pushd $WORKSPACE
curl -fsSL -o ${OPENSSL_NAME}.tar.gz ${OPENSSL_URL}/${OPENSSL_NAME}.tar.gz
check_sha256sum ${OPENSSL_NAME}.tar.gz ${OPENSSL_SHA256}
tar zxf ${OPENSSL_NAME}.tar.gz
pushd ${OPENSSL_NAME}
./config $OPENSSL_BUILD_FLAGS --prefix=$WORKSPACE/ssl --openssldir=$WORKSPACE/ssl
make -j4 > /dev/null
# avoid installing the docs
# https://github.com/openssl/openssl/issues/6685#issuecomment-403838728
make install_sw > /dev/null
popd
rm -rf ${OPENSSL_NAME} ${OPENSSL_NAME}.tar.gz
popd
