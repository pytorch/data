# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data import IterDataPipe

from torch.utils.data.datapipes.iter import (
    Batcher,
    Collator,
    Concater,
    Demultiplexer,
    FileLister,
    FileOpener,
    Filter,
    Forker,
    Grouper,
    IterableWrapper,
    Mapper,
    Multiplexer,
    RoutedDecoder,
    Sampler,
    ShardingFilter,
    Shuffler,
    StreamReader,
    UnBatcher,
    Zipper,
)
from torchdata.datapipes.iter.load.fsspec import (
    FSSpecFileListerIterDataPipe as FSSpecFileLister,
    FSSpecFileOpenerIterDataPipe as FSSpecFileOpener,
    FSSpecSaverIterDataPipe as FSSpecSaver,
)
from torchdata.datapipes.iter.load.iopath import (
    IoPathFileListerIterDataPipe as IoPathFileLister,
    IoPathFileOpenerIterDataPipe as IoPathFileOpener,
    IoPathSaverIterDataPipe as IoPathSaver,
)

from torchdata.datapipes.iter.load.online import (
    GDriveReaderDataPipe as GDriveReader,
    HTTPReaderIterDataPipe as HttpReader,
    OnlineReaderIterDataPipe as OnlineReader,
)
from torchdata.datapipes.iter.transform.bucketbatcher import BucketBatcherIterDataPipe as BucketBatcher
from torchdata.datapipes.iter.transform.flatmap import FlatMapperIterDataPipe as FlatMapper
from torchdata.datapipes.iter.util.cacheholder import (
    EndOnDiskCacheHolderIterDataPipe as EndOnDiskCacheHolder,
    InMemoryCacheHolderIterDataPipe as InMemoryCacheHolder,
    OnDiskCacheHolderIterDataPipe as OnDiskCacheHolder,
)
from torchdata.datapipes.iter.util.combining import (
    IterKeyZipperIterDataPipe as IterKeyZipper,
    MapKeyZipperIterDataPipe as MapKeyZipper,
)
from torchdata.datapipes.iter.util.cycler import CyclerIterDataPipe as Cycler
from torchdata.datapipes.iter.util.dataframemaker import (
    DataFrameMakerIterDataPipe as DataFrameMaker,
    ParquetDFLoaderIterDataPipe as ParquetDataFrameLoader,
)
from torchdata.datapipes.iter.util.extractor import ExtractorIterDataPipe as Extractor
from torchdata.datapipes.iter.util.hashchecker import HashCheckerIterDataPipe as HashChecker
from torchdata.datapipes.iter.util.header import HeaderIterDataPipe as Header
from torchdata.datapipes.iter.util.indexadder import (
    EnumeratorIterDataPipe as Enumerator,
    IndexAdderIterDataPipe as IndexAdder,
)
from torchdata.datapipes.iter.util.jsonparser import JsonParserIterDataPipe as JsonParser
from torchdata.datapipes.iter.util.paragraphaggregator import ParagraphAggregatorIterDataPipe as ParagraphAggregator
from torchdata.datapipes.iter.util.plain_text_reader import (
    CSVDictParserIterDataPipe as CSVDictParser,
    CSVParserIterDataPipe as CSVParser,
    LineReaderIterDataPipe as LineReader,
)
from torchdata.datapipes.iter.util.rararchiveloader import RarArchiveLoaderIterDataPipe as RarArchiveLoader
from torchdata.datapipes.iter.util.rows2columnar import Rows2ColumnarIterDataPipe as Rows2Columnar
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe as SampleMultiplexer
from torchdata.datapipes.iter.util.saver import SaverIterDataPipe as Saver
from torchdata.datapipes.iter.util.tararchivereader import TarArchiveReaderIterDataPipe as TarArchiveReader
from torchdata.datapipes.iter.util.unzipper import UnZipperIterDataPipe as UnZipper
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
    "DataFrameMaker",
    "Demultiplexer",
    "EndOnDiskCacheHolder",
    "Enumerator",
    "Extractor",
    "FSSpecFileLister",
    "FSSpecFileOpener",
    "FSSpecSaver",
    "FileLister",
    "FileOpener",
    "Filter",
    "FlatMapper",
    "Forker",
    "GDriveReader",
    "Grouper",
    "HashChecker",
    "Header",
    "HttpReader",
    "InMemoryCacheHolder",
    "IndexAdder",
    "IoPathFileLister",
    "IoPathFileOpener",
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
    "ParquetDataFrameLoader",
    "RarArchiveLoader",
    "RoutedDecoder",
    "Rows2Columnar",
    "SampleMultiplexer",
    "Sampler",
    "Saver",
    "ShardingFilter",
    "Shuffler",
    "StreamReader",
    "TarArchiveReader",
    "UnBatcher",
    "UnZipper",
    "XzFileReader",
    "ZipArchiveReader",
    "Zipper",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
