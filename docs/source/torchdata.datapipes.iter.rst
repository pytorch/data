Iterable-style DataPipes
==========================

.. currentmodule:: torchdata.datapipes.iter

An iterable-style dataset is an instance of a subclass of IterableDataset that implements the ``__iter__()`` protocol,
and represents an iterable over data samples. This type of datasets is particularly suitable for cases where random
reads are expensive or even improbable, and where the batch size depends on the fetched data.

For example, such a dataset, when called ``iter(iterdatapipe)``, could return a stream of data reading from a database,
a remote server, or even logs generated in real time.

This is an updated version of ``IterableDataset`` in ``torch``.

.. autoclass:: IterDataPipe

We have three types of Iterable DataPipes:

1. Load - help you interact with the file systems or online databases (e.g. FileOpener, GDriveReader)

2. Transform - transform elements within DataPipes (e.g. batching, shuffling)

3. Utility - utility functions (e.g. caching, CSV parsing, filtering)

These DataPipes can be invoked in two ways, using the class constructor or applying their functional form onto
an existing `IterDataPipe` (available to most but not all DataPipes).

.. code:: python

    import torchdata.datapipes.iter import IterableWrapper, Mapper

    dp = IterableWrapper(range(10))
    map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
    map_dp_2 = dp.map(lambda x: x + 1)  # Using functional form
    list(map_dp_1)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    list(map_dp_2)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Load DataPipes
-------------------------

These DataPipes help you interact with the file systems or online databases (e.g. FileOpener, GDriveReader).

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    FSSpecFileLister
    FSSpecFileOpener
    FSSpecSaver
    FileLister
    FileOpener
    GDriveReader
    HttpReader
    IoPathFileLister
    IoPathFileOpener
    OnlineReader
    ParquetDataFrameLoader


Transform DataPipes
-------------------------

These DataPipes transform elements within DataPipes (e.g. batching, shuffling).

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Batcher
    BucketBatcher
    Shuffler

Utility DataPipes
-------------------------

These DataPipes provide utility functions (e.g. caching, CSV parsing, filtering).

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    CSVDictParser
    CSVParser
    Collator
    Concater
    Cycler
    DataFrameMaker
    Demultiplexer
    EndOnDiskCacheHolder
    Enumerator
    Extractor
    Filter
    FlatMapper
    Forker
    Grouper
    HashChecker
    Header
    InMemoryCacheHolder
    IndexAdder
    IoPathSaver
    IterDataPipe
    IterKeyZipper
    IterableWrapper
    JsonParser
    LineReader
    MapKeyZipper
    Mapper
    Multiplexer
    OnDiskCacheHolder
    ParagraphAggregator
    RarArchiveLoader
    RoutedDecoder
    Rows2Columnar
    SampleMultiplexer
    Sampler
    Saver
    ShardingFilter
    StreamReader
    TarArchiveReader
    UnBatcher
    XzFileReader
    ZipArchiveReader
    Zipper
