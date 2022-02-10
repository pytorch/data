Map-style DataPipes
===========================

.. currentmodule:: torchdata.datapipes.map

A map-style dataset is one that implements the ``__getitem__()`` and ``__len__()`` protocols, and represents a map
from (possibly non-integral) indices/keys to data samples.

.. autoclass:: MapDataPipe

For example, such a dataset, when accessed with ``mapdatapipe[idx]``, could read the ``idx``-th image and its
corresponding label from a folder on the disk.

These DataPipes can be invoked in two ways, using the class constructor or applying their functional form onto
an existing `MapDataPipe` (available to most but not all DataPipes).

.. code:: python

    from torchdata.datapipes.map import SequenceWrapper, Mapper

    dp = SequenceWrapper(range(10))
    map_dp_1 = dp.map(lambda x: x + 1)
    list(map_dp_1)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    map_dp_2 = Mapper(dp, lambda x: x + 1)
    list(map_dp_2)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


DataPipes
-------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: datapipe.rst

    Batcher
    Concater
    IterToMapConverter
    Mapper
    SequenceWrapper
    Shuffler
    Zipper
