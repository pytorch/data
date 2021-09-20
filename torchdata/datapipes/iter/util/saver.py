# Copyright (c) Facebook, Inc. and its affiliates.
from io import IOBase
from typing import Any, Callable, Tuple

from torch.utils.data import IterDataPipe, functional_datapipe

from torchdata.datapipes.utils.common import _default_filepath_fn


@functional_datapipe("save_to_disk")
class SaverIterDataPipe(IterDataPipe):
    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[Any, IOBase]],
        mode: str = "wb",
        filepath_fn: Callable = _default_filepath_fn,
    ):
        self.source_datapipe = source_datapipe
        self.mode = mode
        self.fn = filepath_fn

    def __iter__(self):
        for meta, data in self.source_datapipe:
            filepath = self.fn(meta)
            with open(filepath, self.mode) as f:
                f.write(data)
            yield filepath

    def __len__(self):
        return len(self.source_datapipe)
