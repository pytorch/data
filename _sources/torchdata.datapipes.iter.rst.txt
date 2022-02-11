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

1. IO - DESCRIPTION

2. Archive - DESCRIPTION

3. Mapping - DESCRIPTION

4. Selecting - DESCRIPTION

5. Augmenting - DESCRIPTION

6. Combinatorial - DESCRIPTION

7. Text - DESCRIPTION

8. Grouping - DESCRIPTION

9. Combining/Splitting -DESCRIPTION

10. Others - DESCRIPTION

IO DataPipes
-------------------------

These DataPipes help interacting with the file systems or online databases (e.g. downloading, opening,
saving files, and listing the files in directories).

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
    IoPathSaver
    OnlineReader
    ParquetDataFrameLoader
    Saver

Archive DataPipes
-------------------------

These DataPipes help opening and decompressing archive files.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Extractor
    RarArchiveLoader
    TarArchiveReader
    XzFileReader
    ZipArchiveReader

Mapping DataPipes
-------------------------

These DataPipes transform elements within DataPipes (e.g. batching, shuffling).

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    FlatMapper
    Mapper

Selecting DataPipes
-------------------------

These DataPipes helps you select specific samples.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Filter
    Header

Augmenting DataPipes
-----------------------------
These DataPipes help to augment your samples.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Cycler
    Enumerator
    IndexAdder

Combinatorial DataPipes
-----------------------------
These DataPipes help to perform combinatorial operations.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Sampler
    Shuffler

Text DataPipes
-----------------------------
These DataPipes help you parse and read text files.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    CSVDictParser
    CSVParser
    JsonParser
    LineReader
    ParagraphAggregator
    RoutedDecoder
    Rows2Columnar
    StreamReader

Grouping DataPipes
-----------------------------
These DataPipes have you group samples within a DataPipe.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Batcher
    BucketBatcher
    Collator
    Grouper
    UnBatcher

Combining/Spliting DataPipes
-----------------------------
These tend to involve multiple DataPipes and help combining them or spliting one to many.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Concater
    Demultiplexer
    Forker
    IterKeyZipper
    MapKeyZipper
    Multiplexer
    SampleMultiplexer
    UnZipper
    Zipper

Other DataPipes
-------------------------
A miscellaneous set of DataPipes with different functionalities.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    DataFrameMaker
    EndOnDiskCacheHolder
    HashChecker
    InMemoryCacheHolder
    IterableWrapper
    OnDiskCacheHolder
    ShardingFilter
