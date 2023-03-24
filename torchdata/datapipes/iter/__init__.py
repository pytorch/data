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
    BatchAsyncMapperIterDataPipe as BatchAsyncMapper,
    BatchMapperIterDataPipe as BatchMapper,
    DropperIterDataPipe as Dropper,
    FlatMapperIterDataPipe as FlatMapper,
    FlattenIterDataPipe as Flattener,
    ShuffledFlatMapperIterDataPipe as ShuffledFlatMapper,
    SliceIterDataPipe as Slicer,
    ThreadPoolMapperIterDataPipe as ThreadPoolMapper,
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
    RoundRobinDemultiplexerIterDataPipe as RoundRobinDemultiplexer,
    UnZipperIterDataPipe as UnZipper,
)
from torchdata.datapipes.iter.util.cycler import CyclerIterDataPipe as Cycler, RepeaterIterDataPipe as Repeater
from torchdata.datapipes.iter.util.dataframemaker import (
    DataFrameMakerIterDataPipe as DataFrameMaker,
    ParquetDFLoaderIterDataPipe as ParquetDataFrameLoader,
)
from torchdata.datapipes.iter.util.decompressor import (
    DecompressorIterDataPipe as Decompressor,
    ExtractorIterDataPipe as Extractor,
)
from torchdata.datapipes.iter.util.distributed import FullSyncIterDataPipe as FullSync
from torchdata.datapipes.iter.util.hashchecker import HashCheckerIterDataPipe as HashChecker
from torchdata.datapipes.iter.util.header import HeaderIterDataPipe as Header, LengthSetterIterDataPipe as LengthSetter
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
from torchdata.datapipes.iter.util.prefetcher import (
    PinMemoryIterDataPipe as PinMemory,
    PrefetcherIterDataPipe as Prefetcher,
)
from torchdata.datapipes.iter.util.randomsplitter import RandomSplitterIterDataPipe as RandomSplitter
from torchdata.datapipes.iter.util.rararchiveloader import RarArchiveLoaderIterDataPipe as RarArchiveLoader
from torchdata.datapipes.iter.util.rows2columnar import Rows2ColumnarIterDataPipe as Rows2Columnar
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe as SampleMultiplexer
from torchdata.datapipes.iter.util.saver import SaverIterDataPipe as Saver
from torchdata.datapipes.iter.util.shardexpander import ShardExpanderIterDataPipe as ShardExpander
from torchdata.datapipes.iter.util.sharding import (
    ShardingRoundRobinDispatcherIterDataPipe as ShardingRoundRobinDispatcher,
)
from torchdata.datapipes.iter.util.tararchiveloader import TarArchiveLoaderIterDataPipe as TarArchiveLoader
from torchdata.datapipes.iter.util.tfrecordloader import (
    TFRecordExample,
    TFRecordExampleSpec,
    TFRecordLoaderIterDataPipe as TFRecordLoader,
)
from torchdata.datapipes.iter.util.webdataset import WebDatasetIterDataPipe as WebDataset
from torchdata.datapipes.iter.util.xzfileloader import XzFileLoaderIterDataPipe as XzFileLoader
from torchdata.datapipes.iter.util.zip_longest import ZipperLongestIterDataPipe as ZipperLongest
from torchdata.datapipes.iter.util.ziparchiveloader import ZipArchiveLoaderIterDataPipe as ZipArchiveLoader
from torchdata.datapipes.map.util.converter import MapToIterConverterIterDataPipe as MapToIterConverter

__all__ = [
    "AISFileLister",
    "AISFileLoader",
    "BatchAsyncMapper",
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
    "Flattener",
    "Forker",
    "FullSync",
    "GDriveReader",
    "Grouper",
    "HashChecker",
    "Header",
    "HttpReader",
    "HuggingFaceHubReader",
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
    "LengthSetter",
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
    "PinMemory",
    "Prefetcher",
    "RandomSplitter",
    "RarArchiveLoader",
    "Repeater",
    "RoundRobinDemultiplexer",
    "RoutedDecoder",
    "Rows2Columnar",
    "S3FileLister",
    "S3FileLoader",
    "SampleMultiplexer",
    "Sampler",
    "Saver",
    "ShardExpander",
    "ShardingFilter",
    "ShardingRoundRobinDispatcher",
    "ShuffledFlatMapper",
    "Shuffler",
    "Slicer",
    "StreamReader",
    "TFRecordLoader",
    "TarArchiveLoader",
    "ThreadPoolMapper",
    "UnBatcher",
    "UnZipper",
    "WebDataset",
    "XzFileLoader",
    "ZipArchiveLoader",
    "Zipper",
    "ZipperLongest",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
