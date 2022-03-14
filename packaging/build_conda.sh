#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export UPLOAD_CHANNEL="test"

export BUILD_TYPE="conda"
export NO_CUDA_PACKAGE=1
export CU_VERSION=cpu

export SOURCE_ROOT_DIR="$PWD"
setup_env $1
setup_conda_pytorch_constraint

mkdir -p conda-bld
conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload --output-folder conda-bld --python "$PYTHON_VERSION" packaging/torchdata
