# Copyright (c) Facebook, Inc. and its affiliates.

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

from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

from torch.utils.data import DataChunk, IterableDataset
from torch.utils.data._typing import _DataPipeMeta
from torchdata.datapipes.map import MapDataPipe

########################################################################################################################
# The part below is generated by parsing through the Python files where IterDataPipes are defined.
# This base template ("__init__.pyi.in") is generated from mypy stubgen with minimal editing for code injection
# The output file will be "__init__.pyi".
# Note that, for mypy, .pyi file takes precedent over .py file, such that we must define the interface for other
# classes/objects here, even though we are not injecting extra code into them at the moment.

from .util.extractor import CompressionType

try:
    import torcharrow
except ImportError:
    torcharrow = None

T_co = TypeVar("T_co", covariant=True)

class IterDataPipe(IterableDataset[T_co], metaclass=_DataPipeMeta):
    functions: Dict[str, Callable] = ...
    reduce_ex_hook: Optional[Callable] = ...
    getstate_hook: Optional[Callable] = ...
    def __getattr__(self, attribute_name: Any): ...
    @classmethod
    def register_function(cls, function_name: Any, function: Any) -> None: ...
    @classmethod
    def register_datapipe_as_function(
        cls, function_name: Any, cls_to_register: Any, enable_df_api_tracing: bool = ...
    ): ...
    def __getstate__(self): ...
    def __reduce_ex__(self, *args: Any, **kwargs: Any): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn: Any) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn: Any) -> None: ...
    # Functional form of 'BatcherIterDataPipe'
    def batch(self, batch_size: int, drop_last: bool = False, wrapper_class=DataChunk) -> IterDataPipe: ...
    # Functional form of 'CollatorIterDataPipe'
    def collate(self, collate_fn: Callable = ...) -> IterDataPipe: ...
    # Functional form of 'ConcaterIterDataPipe'
    def concat(self, *datapipes: IterDataPipe) -> IterDataPipe: ...
    # Functional form of 'DemultiplexerIterDataPipe'
    def demux(
        self,
        num_instances: int,
        classifier_fn: Callable[[T_co], Optional[int]],
        drop_none: bool = False,
        buffer_size: int = 1000,
    ) -> List[IterDataPipe]: ...
    # Functional form of 'FilterIterDataPipe'
    def filter(self, filter_fn: Callable, drop_empty_batches: bool = True) -> IterDataPipe: ...
    # Functional form of 'ForkerIterDataPipe'
    def fork(self, num_instances: int, buffer_size: int = 1000) -> List[IterDataPipe]: ...
    # Functional form of 'GrouperIterDataPipe'
    def groupby(
        self,
        group_key_fn: Callable,
        *,
        buffer_size: int = 10000,
        group_size: Optional[int] = None,
        guaranteed_group_size: Optional[int] = None,
        drop_remaining: bool = False,
    ) -> IterDataPipe: ...
    # Functional form of 'MapperIterDataPipe'
    def map(self, fn: Callable, input_col=None, output_col=None) -> IterDataPipe: ...
    # Functional form of 'MultiplexerIterDataPipe'
    def mux(self, *datapipes) -> IterDataPipe: ...
    # Functional form of 'RoutedDecoderIterDataPipe'
    def routed_decode(self, *handlers: Callable, key_fn: Callable = ...) -> IterDataPipe: ...
    # Functional form of 'ShardingFilterIterDataPipe'
    def sharding_filter(self) -> IterDataPipe: ...
    # Functional form of 'ShufflerIterDataPipe'
    def shuffle(self, *, default: bool = True, buffer_size: int = 10000, unbatch_level: int = 0) -> IterDataPipe: ...
    # Functional form of 'UnBatcherIterDataPipe'
    def unbatch(self, unbatch_level: int = 1) -> IterDataPipe: ...
    # Functional form of 'ZipperIterDataPipe'
    def zip(self, *datapipes: IterDataPipe) -> IterDataPipe: ...
    # Functional form of 'IndexAdderIterDataPipe'
    def add_index(self, index_name: str = "index") -> IterDataPipe: ...
    # Functional form of 'BucketBatcherIterDataPipe'
    def bucketbatch(
        self,
        batch_size: int,
        drop_last: bool = False,
        batch_num: int = 100,
        bucket_num: int = 1,
        sort_key: Optional[Callable] = None,
        in_batch_shuffle: bool = True,
    ) -> IterDataPipe: ...
    # Functional form of 'HashCheckerIterDataPipe'
    def check_hash(self, hash_dict: Dict[str, str], hash_type: str = "sha256", rewind: bool = True) -> IterDataPipe: ...
    # Functional form of 'CyclerIterDataPipe'
    def cycle(self, count: Optional[int] = None) -> IterDataPipe: ...
    # Functional form of 'DataFrameMakerIterDataPipe'
    def dataframe(
        self, dataframe_size: int = 1000, dtype=None, columns: Optional[List[str]] = None, device: str = ""
    ) -> torcharrow.DataFrame: ...
    # Functional form of 'EndOnDiskCacheHolderIterDataPipe'
    def end_caching(self, mode="wb", filepath_fn=None, *, same_filepath_fn=False, skip_read=False) -> IterDataPipe: ...
    # Functional form of 'EnumeratorIterDataPipe'
    def enumerate(self, starting_index: int = 0) -> IterDataPipe: ...
    # Functional form of 'ExtractorIterDataPipe'
    def extract(self, file_type: Optional[Union[str, CompressionType]] = None) -> IterDataPipe: ...
    # Functional form of 'FlatMapperIterDataPipe'
    def flatmap(self, fn: Callable) -> IterDataPipe: ...
    # Functional form of 'HeaderIterDataPipe'
    def header(self, limit: int = 10) -> IterDataPipe: ...
    # Functional form of 'InMemoryCacheHolderIterDataPipe'
    def in_memory_cache(self, size: Optional[int] = None) -> IterDataPipe: ...
    # Functional form of 'ParagraphAggregatorIterDataPipe'
    def lines_to_paragraphs(self, joiner: Callable = ...) -> IterDataPipe: ...
    # Functional form of 'RarArchiveLoaderIterDataPipe'
    def load_from_rar(self, *, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'ParquetDFLoaderIterDataPipe'
    def load_parquet_as_df(
        self, dtype=None, columns: Optional[List[str]] = None, device: str = "", use_threads: bool = False
    ) -> IterDataPipe: ...
    # Functional form of 'OnDiskCacheHolderIterDataPipe'
    def on_disk_cache(
        self,
        filepath_fn: Optional[Callable] = None,
        hash_dict: Dict[str, str] = None,
        hash_type: str = "sha256",
        extra_check_fn: Optional[Callable[[str], bool]] = None,
    ) -> IterDataPipe: ...
    # Functional form of 'FSSpecFileOpenerIterDataPipe'
    def open_file_by_fsspec(self, mode: str = "r") -> IterDataPipe: ...
    # Functional form of 'IoPathFileOpenerIterDataPipe'
    def open_file_by_iopath(self, mode: str = "r", pathmgr=None) -> IterDataPipe: ...
    # Functional form of 'CSVParserIterDataPipe'
    def parse_csv(
        self,
        *,
        skip_lines: int = 0,
        decode: bool = True,
        encoding: str = "utf-8",
        errors: str = "ignore",
        return_path: bool = False,
        **fmtparams,
    ) -> IterDataPipe: ...
    # Functional form of 'CSVDictParserIterDataPipe'
    def parse_csv_as_dict(
        self,
        *,
        skip_lines: int = 0,
        decode: bool = True,
        encoding: str = "utf-8",
        errors: str = "ignore",
        return_path: bool = False,
        **fmtparams,
    ) -> IterDataPipe: ...
    # Functional form of 'JsonParserIterDataPipe'
    def parse_json_files(self, **kwargs) -> IterDataPipe: ...
    # Functional form of 'TarArchiveReaderIterDataPipe'
    def read_from_tar(self, mode: str = "r:*", length: int = -1) -> IterDataPipe: ...
    # Functional form of 'XzFileReaderIterDataPipe'
    def read_from_xz(self, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'ZipArchiveReaderIterDataPipe'
    def read_from_zip(self, length: int = -1) -> IterDataPipe: ...
    # Functional form of 'LineReaderIterDataPipe'
    def readlines(
        self,
        *,
        skip_lines: int = 0,
        strip_newline: bool = True,
        decode: bool = False,
        encoding="utf-8",
        errors: str = "ignore",
        return_path: bool = True,
    ) -> IterDataPipe: ...
    # Functional form of 'Rows2ColumnarIterDataPipe'
    def rows2columnar(self, column_names: List[str] = None) -> IterDataPipe: ...
    # Functional form of 'FSSpecSaverIterDataPipe'
    def save_by_fsspec(self, mode: str = "w", filepath_fn: Optional[Callable] = None) -> IterDataPipe: ...
    # Functional form of 'IoPathSaverIterDataPipe'
    def save_by_iopath(
        self, mode: str = "w", filepath_fn: Optional[Callable] = None, *, pathmgr=None
    ) -> IterDataPipe: ...
    # Functional form of 'SaverIterDataPipe'
    def save_to_disk(self, mode: str = "w", filepath_fn: Optional[Callable] = None) -> IterDataPipe: ...
    # Functional form of 'UnZipperIterDataPipe'
    def unzip(
        self, sequence_length: int, buffer_size: int = 1000, columns_to_skip: Optional[Sequence[int]] = None
    ) -> List[IterDataPipe]: ...
    # Functional form of 'IterKeyZipperIterDataPipe'
    def zip_with_iter(
        self,
        ref_datapipe: IterDataPipe,
        key_fn: Callable,
        ref_key_fn: Optional[Callable] = None,
        keep_key: bool = False,
        buffer_size: int = 10000,
        merge_fn: Optional[Callable] = None,
    ) -> IterDataPipe: ...
    # Functional form of 'MapKeyZipperIterDataPipe'
    def zip_with_map(
        self, map_datapipe: MapDataPipe, key_fn: Callable, merge_fn: Optional[Callable] = None
    ) -> IterDataPipe: ...
