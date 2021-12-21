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
    ShardingFilter,
)
# TODO: import ShardingFilter directly from torch.utils.data.datapipes.iter as soon as it is exposed.
#  https://github.com/pytorch/pytorch/pull/69844 was merged, but is not yet available in the nightly releases
from torch.utils.data.datapipes.iter.grouping import ShardingFilterIterDataPipe as ShardingFilter

from torchdata.datapipes.iter.load.online import (
    OnlineReaderIterDataPipe as OnlineReader,
    HTTPReaderIterDataPipe as HttpReader,
    GDriveReaderDataPipe as GDriveReader,
)
from torchdata.datapipes.iter.load.iopath import (
    IoPathFileListerIterDataPipe as IoPathFileLister,
    IoPathFileLoaderIterDataPipe as IoPathFileLoader,
    IoPathSaverIterDataPipe as IoPathSaver,
)
from torchdata.datapipes.iter.load.fsspec import (
    FSSpecFileListerIterDataPipe as FSSpecFileLister,
    FSSpecFileOpenerIterDataPipe as FSSpecFileOpener,
    FSSpecSaverIterDataPipe as FSSpecSaver,
)
from torchdata.datapipes.iter.load.s3io import (
    S3FileListerIterDataPipe as S3FileLister,
    S3FileLoaderIterDataPipe as S3FileLoader,
)
from torchdata.datapipes.iter.transform.bucketbatcher import BucketBatcherIterDataPipe as BucketBatcher
from torchdata.datapipes.iter.util.cacheholder import (
    EndOnDiskCacheHolderIterDataPipe as EndOnDiskCacheHolder,
    InMemoryCacheHolderIterDataPipe as InMemoryCacheHolder,
    OnDiskCacheHolderIterDataPipe as OnDiskCacheHolder,
)
from torchdata.datapipes.iter.util.indexadder import (
    EnumeratorIterDataPipe as Enumerator,
    IndexAdderIterDataPipe as IndexAdder,
)
from torchdata.datapipes.iter.util.combining import (
    IterKeyZipperIterDataPipe as IterKeyZipper,
    MapKeyZipperIterDataPipe as MapKeyZipper,
)
from torchdata.datapipes.iter.util.plain_text_reader import (
    LineReaderIterDataPipe as LineReader,
    CSVDictParserIterDataPipe as CSVDictParser,
    CSVParserIterDataPipe as CSVParser,
)
from torchdata.datapipes.iter.util.cycler import CyclerIterDataPipe as Cycler
from torchdata.datapipes.iter.util.extractor import ExtractorIterDataPipe as Extractor
from torchdata.datapipes.iter.util.hashchecker import HashCheckerIterDataPipe as HashChecker
from torchdata.datapipes.iter.util.header import HeaderIterDataPipe as Header
from torchdata.datapipes.iter.util.jsonparser import JsonParserIterDataPipe as JsonParser
from torchdata.datapipes.iter.util.paragraphaggregator import ParagraphAggregatorIterDataPipe as ParagraphAggregator
from torchdata.datapipes.iter.util.rar_archive_loader import RarArchiveLoaderIterDataPipe as RarArchiveLoader
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
    "EndOnDiskCacheHolder",
    "Enumerator",
    "Extractor",
    "FSSpecFileLister",
    "FSSpecFileOpener",
    "FSSpecSaver",
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
    "IoPathSaver",
    "IterDataPipe",
    "IterKeyZipper",
    "IterableWrapper",
    "JsonParser",
    "LineReader",
    "MapKeyZipper",
    "Mapper",
    "Multiplexer",
    "OnDiskCacheHolder",
    "OnlineReader",
    "ParagraphAggregator",
    "RarArchiveLoader",
    "RoutedDecoder",
    "Rows2Columnar",
    "S3FileLister",
    "S3FileLoader",
    "SampleMultiplexer",
    "Sampler",
    "Saver",
    "ShardingFilter",
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
