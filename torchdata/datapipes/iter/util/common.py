# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata.datapipes.iter import Mapper


class MapTemplateIterDataPipe(Mapper):
    def __init__(self, source_datapipe, input_col=None, output_col=None) -> None:
        fn = self.map
        super().__init__(source_datapipe, fn=fn, input_col=input_col, output_col=output_col)

    def map(self, *args, **kwargs):
        raise NotImplementedError
