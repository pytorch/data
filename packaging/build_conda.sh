#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export CU_VERSION=cpu
export NO_CUDA_PACKAGE=1
export BUILD_TYPE="conda"

if [[ "$PYTHON_VERSION" == "3.11" ]]; then
  export CONDA_CHANNEL_FLAGS="${$CONDA_CHANNEL_FLAGS} -c malfet"
fi

export SOURCE_ROOT_DIR="$PWD"
setup_env
setup_conda_pytorch_constraint

mkdir -p conda-bld
conda build \
  -c defaults \
  $CONDA_CHANNEL_FLAGS \
  --no-anaconda-upload \
  --output-folder conda-bld \
  --python "$PYTHON_VERSION" \
  packaging/torchdata
