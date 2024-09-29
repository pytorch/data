# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

class DataSource(ABC):
    @abstractmethod
    def get_bundle(i):
        ...
    
    @abstractmethod    
    def get_num_bundles():
        ...        
    
    def get_samples_per_bundle():
        raise NotImplementedError("get_samples_per_bundle is not implemented")

    def prefetch(i):
        raise NotImplementedError("prefetch is not implemented")
