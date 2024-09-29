# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .datasource import DataSource
from .bundle import Bundle

class DataSourceReader():
    def __init__(self, datasource: DataSource, shuffle=False, sampler=None):
        raise NotImplementedError("DataSourceReader is not implemented")

    def get_sampler(self):
        raise NotImplementedError("get_sampler is not implemented")
    
    def to_map_dataset(self):
        raise NotImplementedError("to_map_dataset is not implemented")

    def to_iterable_dataset(self):
        raise NotImplementedError("to_iterable_Dataset is not implemented")
