# Copyright (c) Facebook, Inc. and its affiliates.
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
from torchdata.datapipes.iter.util.rows2columnar import Rows2ColumnarIterDataPipe as Rows2Columnar
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe as SampleMultiplexer
from torchdata.datapipes.iter.util.saver import SaverIterDataPipe as Saver
from torchdata.datapipes.iter.util.tararchivereader import TarArchiveReaderIterDataPipe as TarArchiveReader
from torchdata.datapipes.iter.util.xzfilereader import XzFileReaderIterDataPipe as XzFileReader
from torchdata.datapipes.iter.util.ziparchivereader import ZipArchiveReaderIterDataPipe as ZipArchiveReader

###############################################################################
# Reference From PyTorch Core
###############################################################################
from torch.utils.data.datapipes.iter import IterableWrapper

__all__ = [
    "BucketBatcher",
    "CSVDictParser",
    "CSVParser",
    "Cycler",
    "GDriveReader",
    "HashChecker",
    "Header",
    "HttpReader",
    "InMemoryCacheHolder",
    "IndexAdder",
    "IoPathFileLister",
    "IoPathFileLoader",
    "IterableWrapper",
    "JsonParser",
    "KeyZipper",
    "LineReader",
    "OnDiskCacheHolder",
    "OnlineReader",
    "ParagraphAggregator",
    "Rows2Columnar",
    "SampleMultiplexer",
    "Saver",
    "TarArchiveReader",
    "XzFileReader",
    "ZipArchiveReader",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
