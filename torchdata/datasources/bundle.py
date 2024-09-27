# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

class Bundle(ABC):
    @abstractmethod
    def __iter__():
        ...
    
    def get_sample(i):
        raise NotImplementedError("get_sample is not implemented")
    
    def get_num_samples():
        raise NotImplementedError
   
    @abstractmethod
    def state_dict():
        ...
    
    @abstractmethod
    def load_state_dict():
        ...
