#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export CU_VERSION=cpu
export NO_CUDA_PACKAGE=1
export BUILD_TYPE="conda"

export SOURCE_ROOT_DIR="$PWD"
setup_env
setup_conda_pytorch_constraint

mkdir -p conda-bld
conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload --output-folder conda-bld --python "$PYTHON_VERSION" packaging/torchdata
