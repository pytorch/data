# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

###############################################################################
# Reference From PyTorch Core
###############################################################################
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
from torchdata.datapipes.iter.load.aisio import (
    AISFileListerIterDataPipe as AISFileLister,
    AISFileLoaderIterDataPipe as AISFileLoader,
)

###############################################################################
# TorchData
###############################################################################
from torchdata.datapipes.iter.load.fsspec import (
    FSSpecFileListerIterDataPipe as FSSpecFileLister,
    FSSpecFileOpenerIterDataPipe as FSSpecFileOpener,
    FSSpecSaverIterDataPipe as FSSpecSaver,
)

from torchdata.datapipes.iter.load.huggingface import HuggingFaceHubReaderIterDataPipe as HuggingFaceHubReader

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
from torchdata.datapipes.iter.load.s3io import (
    S3FileListerIterDataPipe as S3FileLister,
    S3FileLoaderIterDataPipe as S3FileLoader,
)
from torchdata.datapipes.iter.transform.bucketbatcher import (
    BucketBatcherIterDataPipe as BucketBatcher,
    InBatchShufflerIterDataPipe as InBatchShuffler,
    MaxTokenBucketizerIterDataPipe as MaxTokenBucketizer,
)
from torchdata.datapipes.iter.transform.callable import (
    BatchMapperIterDataPipe as BatchMapper,
    DropperIterDataPipe as Dropper,
    FlatMapperIterDataPipe as FlatMapper,
    ISliceIterDataPipe as ISlicer,
)
from torchdata.datapipes.iter.util.bz2fileloader import Bz2FileLoaderIterDataPipe as Bz2FileLoader
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
from torchdata.datapipes.iter.util.decompressor import (
    DecompressorIterDataPipe as Decompressor,
    ExtractorIterDataPipe as Extractor,
)
from torchdata.datapipes.iter.util.hashchecker import HashCheckerIterDataPipe as HashChecker
from torchdata.datapipes.iter.util.header import HeaderIterDataPipe as Header
from torchdata.datapipes.iter.util.indexadder import (
    EnumeratorIterDataPipe as Enumerator,
    IndexAdderIterDataPipe as IndexAdder,
)
from torchdata.datapipes.iter.util.jsonparser import JsonParserIterDataPipe as JsonParser
from torchdata.datapipes.iter.util.mux_longest import MultiplexerLongestIterDataPipe as MultiplexerLongest
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
from torchdata.datapipes.iter.util.tararchiveloader import (
    TarArchiveLoaderIterDataPipe as TarArchiveLoader,
    TarArchiveReaderIterDataPipe as TarArchiveReader,
)
from torchdata.datapipes.iter.util.tfrecordloader import (
    TFRecordExample,
    TFRecordExampleSpec,
    TFRecordLoaderIterDataPipe as TFRecordLoader,
)
from torchdata.datapipes.iter.util.unzipper import UnZipperIterDataPipe as UnZipper
from torchdata.datapipes.iter.util.webdataset import WebDatasetIterDataPipe as WebDataset
from torchdata.datapipes.iter.util.xzfileloader import (
    XzFileLoaderIterDataPipe as XzFileLoader,
    XzFileReaderIterDataPipe as XzFileReader,
)
from torchdata.datapipes.iter.util.zip_longest import ZipperLongestIterDataPipe as ZipperLongest
from torchdata.datapipes.iter.util.ziparchiveloader import (
    ZipArchiveLoaderIterDataPipe as ZipArchiveLoader,
    ZipArchiveReaderIterDataPipe as ZipArchiveReader,
)
from torchdata.datapipes.map.util.converter import MapToIterConverterIterDataPipe as MapToIterConverter

__all__ = [
    "AISFileLister",
    "AISFileLoader",
    "BatchMapper",
    "Batcher",
    "BucketBatcher",
    "Bz2FileLoader",
    "CSVDictParser",
    "CSVParser",
    "Collator",
    "Concater",
    "Cycler",
    "DataFrameMaker",
    "Decompressor",
    "Demultiplexer",
    "Dropper",
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
    "HuggingFaceHubReader",
    "ISlicer",
    "InBatchShuffler",
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
    "MapToIterConverter",
    "Mapper",
    "MaxTokenBucketizer",
    "Multiplexer",
    "MultiplexerLongest",
    "OnDiskCacheHolder",
    "OnlineReader",
    "ParagraphAggregator",
    "ParquetDataFrameLoader",
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
    "TFRecordLoader",
    "TarArchiveLoader",
    "TarArchiveReader",
    "UnBatcher",
    "UnZipper",
    "WebDataset",
    "XzFileLoader",
    "XzFileReader",
    "ZipArchiveLoader",
    "ZipArchiveReader",
    "Zipper",
    "ZipperLongest",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)

