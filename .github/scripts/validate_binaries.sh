#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

AWS_ENABLED=1
if [[ ${MATRIX_PACKAGE_TYPE} = "conda" ]]; then

    if [[ "${MATRIX_PYTHON_VERSION}" = "3.11" ]]; then
        PYTORCH_CONDA_CHANNEL="malfet -c ${PYTORCH_CONDA_CHANNEL}"
    fi

    # shellcheck disable=SC2086
    conda install -y torchdata -c ${PYTORCH_CONDA_CHANNEL}
    if [[ ${TARGET_OS:-} = "windows" ]]; then
        AWS_ENABLED=0
    fi
else
    # shellcheck disable=SC2086
    pip install ${PYTORCH_PIP_PREFIX:-} torchdata --extra-index-url "${PYTORCH_PIP_DOWNLOAD_URL}"
fi

case "${AWS_ENABLED}" in
    "0")
        python ./test/smoke_test/smoke_test.py --no-s3
        ;;
    "1")
        python ./test/smoke_test/smoke_test.py
        ;;
    *)
        exit 1
        ;;
esac
