#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

python_nodot="$(echo $PYTHON_VERSION | tr -d '.')"
case $PYTHON_VERSION in
  3.[6-7]*)
    DESIRED_PYTHON="cp${python_nodot}-cp${python_nodot}m"
    ;;
  3.*)
    DESIRED_PYTHON="cp${python_nodot}-cp${python_nodot}"
    ;;
esac

pydir="/opt/python/$DESIRED_PYTHON"
export PATH="$pydir/bin:$PATH"
