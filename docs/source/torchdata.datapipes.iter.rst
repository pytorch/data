
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


We have different types of Iterable DataPipes:

1. Archive - open and decompress archive files of different formats.

2. Augmenting - augment your samples (e.g. adding index, or cycle through indefinitely).

3. Combinatorial - perform combinatorial operations (e.g. sampling, shuffling).

4. Combining/Splitting - interact with multiple DataPipes by combining them or splitting one to many.

5. Grouping - group samples within a DataPipe

6. IO - interacting with the file systems or remote server (e.g. downloading, opening,
   saving files, and listing the files in directories).

7. Mapping - apply the a given function to each element in the DataPipe.

8. Others - perform miscellaneous set of operations.

9. Selecting - select specific samples within a DataPipe.

10. Text - parse, read, and transform text files and data

Archive DataPipes
-------------------------

These DataPipes help opening and decompressing archive files of different formats.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Bz2FileLoader
    Decompressor
    RarArchiveLoader
    TarArchiveLoader
    WebDataset
    XzFileLoader
    ZipArchiveLoader

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
    Repeater

Combinatorial DataPipes
-----------------------------
These DataPipes help to perform combinatorial operations.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    InBatchShuffler
    Sampler
    Shuffler

Combining/Spliting DataPipes
-----------------------------
These tend to involve multiple DataPipes, combining them or splitting one to many.

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
    MultiplexerLongest
    SampleMultiplexer
    UnZipper
    Zipper
    ZipperLongest

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
    MaxTokenBucketizer
    UnBatcher

IO DataPipes
-------------------------

These DataPipes help interacting with the file systems or remote server (e.g. downloading, opening,
saving files, and listing the files in directories).

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    AISFileLister
    AISFileLoader
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
    S3FileLister
    S3FileLoader
    Saver

Mapping DataPipes
-------------------------

These DataPipes apply the a given function to each element in the DataPipe.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    BatchMapper
    FlatMapper
    Mapper

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
    LengthSetter
    MapToIterConverter
    OnDiskCacheHolder
    RandomSplitter
    ShardingFilter

Selecting DataPipes
-------------------------

These DataPipes helps you select specific samples within a DataPipe.

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Filter
    Header
    Dropper
    Slicer
    Flattener

Text DataPipes
-----------------------------
These DataPipes help you parse, read, and transform text files and data.

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
