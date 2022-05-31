#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

OPENSSL_URL="https://www.openssl.org/source/"
OPENSSL_NAME="openssl-1.1.1o"
OPENSSL_SHA256="9384a2b0570dd80358841464677115df785edb941c71211f75076d72fe6b438f"
OPENSSL_BUILD_FLAGS="no-ssl2 no-zlib no-shared no-comp no-dynamic-engine enable-ec_nistp_64_gcc_128"

CURL_URL="https://github.com/curl/curl/releases/download"
CURL_NAME="curl-7.83.1"
CURL_BUILD_FLAGS="--disable-shared"

function check_sha256sum {
    local fname=$1
    local sha256=$2
    echo "${sha256}  ${fname}" > ${fname}.sha256
    sha256sum -c ${fname}.sha256
    rm ${fname}.sha256
}

yum erase -y openssl-devel curl-devel

pushd ${WORKSPACE}

# OpenSSL
curl -fsSL -o ${OPENSSL_NAME}.tar.gz ${OPENSSL_URL}/${OPENSSL_NAME}.tar.gz
check_sha256sum ${OPENSSL_NAME}.tar.gz ${OPENSSL_SHA256}
tar zxf ${OPENSSL_NAME}.tar.gz

pushd ${OPENSSL_NAME}

./config $OPENSSL_BUILD_FLAGS --prefix=${WORKSPACE}/ssl --openssldir=${WORKSPACE}/ssl
make -j4 > /dev/null
# avoid installing the docs
# https://github.com/openssl/openssl/issues/6685#issuecomment-403838728
make install_sw > /dev/null

popd
rm -rf ${OPENSSL_NAME} ${OPENSSL_NAME}.tar.gz

# cURL
curl -fsSL -o ${CURL_NAME}.tar.gz ${CURL_URL}/${CURL_NAME//./_}/${CURL_NAME}.tar.gz
tar zxf ${CURL_NAME}.tar.gz

pushd ${CURL_NAME}

./configure ${CURL_BUILD_FLAGS} --with-openssl=${WORKSPACE}/ssl --prefix=${WORKSPACE}/curl 
make -j4 > /dev/null
make install > /dev/null

popd
rm -rf ${CURL_NAME} ${CURL_NAME}.tar.gz

popd
