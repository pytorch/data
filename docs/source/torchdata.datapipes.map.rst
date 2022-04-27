Map-style DataPipes
===========================

.. currentmodule:: torchdata.datapipes.map

A Map-style DataPipe is one that implements the ``__getitem__()`` and ``__len__()`` protocols, and represents a map
from (possibly non-integral) indices/keys to data samples.

For example, when accessed with ``mapdatapipe[idx]``, could read the ``idx``-th image and its
corresponding label from a folder on the disk.

.. autoclass:: MapDataPipe


Here is the list of available Map-style DataPipes:

DataPipes
-------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Batcher
    Concater
    InMemoryCacheHolder
    Mapper
    MapToIterConverter
    SequenceWrapper
    Shuffler
    UnZipper
    Zipper
