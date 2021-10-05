# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    Collator,
    Mapper,
    Sampler,
    Shuffler,
    Concater,
    Demultiplexer,
    Forker,
    Multiplexer,
    Zipper,
    FileLister,
    FileLoader,
    Batcher,
    Grouper,
    UnBatcher,
    RoutedDecoder,
    Filter,
    StreamReader,
    IterableWrapper,
)

from torchdata.datapipes.iter.load.online import (
    OnlineReaderIterDataPipe as OnlineReader,
    HTTPReaderIterDataPipe as HttpReader,
    GDriveReaderDataPipe as GDriveReader,
)
from torchdata.datapipes.iter.load.iopath import (
    IoPathFileListerIterDataPipe as IoPathFileLister,
    IoPathFileLoaderIterDataPipe as IoPathFileLoader,
)
from torchdata.datapipes.iter.transform.bucketbatcher import BucketBatcherIterDataPipe as BucketBatcher
from torchdata.datapipes.iter.util.cacheholder import (
    InMemoryCacheHolderIterDataPipe as InMemoryCacheHolder,
    OnDiskCacheHolderIterDataPipe as OnDiskCacheHolder,
)
from torchdata.datapipes.iter.util.indexadder import IndexAdderIterDataPipe as IndexAdder
from torchdata.datapipes.iter.util.combining import KeyZipperIterDataPipe as KeyZipper
from torchdata.datapipes.iter.util.csvparser import (
    CSVDictParserIterDataPipe as CSVDictParser,
    CSVParserIterDataPipe as CSVParser,
)
from torchdata.datapipes.iter.util.cycler import CyclerIterDataPipe as Cycler
from torchdata.datapipes.iter.util.hashchecker import HashCheckerIterDataPipe as HashChecker
from torchdata.datapipes.iter.util.header import HeaderIterDataPipe as Header
from torchdata.datapipes.iter.util.jsonparser import JsonParserIterDataPipe as JsonParser
from torchdata.datapipes.iter.util.linereader import LineReaderIterDataPipe as LineReader
from torchdata.datapipes.iter.util.paragraphaggregator import ParagraphAggregatorIterDataPipe as ParagraphAggregator
from torchdata.datapipes.iter.util.rar_archive_reader import RarArchiveReaderIterDataPipe as RarArchiveReader
from torchdata.datapipes.iter.util.rows2columnar import Rows2ColumnarIterDataPipe as Rows2Columnar
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe as SampleMultiplexer
from torchdata.datapipes.iter.util.saver import SaverIterDataPipe as Saver
from torchdata.datapipes.iter.util.tararchivereader import TarArchiveReaderIterDataPipe as TarArchiveReader
from torchdata.datapipes.iter.util.xzfilereader import XzFileReaderIterDataPipe as XzFileReader
from torchdata.datapipes.iter.util.ziparchivereader import ZipArchiveReaderIterDataPipe as ZipArchiveReader

###############################################################################
# Reference From PyTorch Core
###############################################################################

__all__ = [
    "Batcher",
    "BucketBatcher",
    "CSVDictParser",
    "CSVParser",
    "Collator",
    "Concater",
    "Cycler",
    "Demultiplexer",
    "FileLister",
    "FileLoader",
    "Filter",
    "Forker",
    "GDriveReader",
    "Grouper",
    "HashChecker",
    "Header",
    "HttpReader",
    "InMemoryCacheHolder",
    "IndexAdder",
    "IoPathFileLister",
    "IoPathFileLoader",
    "IterDataPipe",
    "IterableWrapper",
    "JsonParser",
    "KeyZipper",
    "LineReader",
    "Mapper",
    "Multiplexer",
    "OnDiskCacheHolder",
    "OnlineReader",
    "ParagraphAggregator",
    "RarArchiveReader",
    "RoutedDecoder",
    "Rows2Columnar",
    "SampleMultiplexer",
    "Sampler",
    "Saver",
    "Shuffler",
    "StreamReader",
    "TarArchiveReader",
    "UnBatcher",
    "XzFileReader",
    "ZipArchiveReader",
    "Zipper",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
