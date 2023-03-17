# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tqdm

from typing import TypeVar, Callable, Iterator

from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

D = TypeVar("D")


@functional_datapipe("show_progress")
class ProgressBar(IterDataPipe):
    def __init__(
        self, datapipe: IterDataPipe[D], *, update_fn: Callable[[D], int], reset_fn: Callable[[D], bool] = None
    ) -> None:
        self.datapipe = datapipe
        self.update_fn = update_fn
        self.reset_fn = reset_fn

    def __iter__(self) -> Iterator[D]:
        reset = True
        for data in self.datapipe:
            if reset:
                progress_bar = tqdm.tqdm(desc=data[0])
                reset = False

            progress_bar.update(self.update_fn(data))

            if self.reset_fn and self.reset_fn(data):
                progress_bar.close()
                reset = True

            yield data
        progress_bar.close()
