# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data import MapDataPipe

from torch.utils.data.datapipes.map import Batcher, Concater, Mapper, SequenceWrapper, Shuffler, Zipper

from torchdata.datapipes.iter.util.converter import IterToMapConverterMapDataPipe as IterToMapConverter
from torchdata.datapipes.map.util.cacheholder import InMemoryCacheHolderMapDataPipe as InMemoryCacheHolder
from torchdata.datapipes.map.util.unzipper import UnZipperMapDataPipe as UnZipper

__all__ = [
    "Batcher",
    "Concater",
    "InMemoryCacheHolder",
    "IterToMapConverter",
    "MapDataPipe",
    "Mapper",
    "SequenceWrapper",
    "Shuffler",
    "UnZipper",
    "Zipper",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
