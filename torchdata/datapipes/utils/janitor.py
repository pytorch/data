# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata.datapipes.utils import StreamWrapper


def janitor(obj):
    """
    Invokes various `obj` cleanup procedures such as:
    - Closing streams
    """
    # TODO(632): We can also release caching locks here to allow filtering
    StreamWrapper.close_streams(obj)
