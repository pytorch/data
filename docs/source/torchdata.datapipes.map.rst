Map-style DataPipes
===========================

.. currentmodule:: torchdata.datapipes.map

A Map-style DataPipe is one that implements the ``__getitem__()`` and ``__len__()`` protocols, and represents a map
from (possibly non-integral) indices/keys to data samples. This is a close equivalent of ``Dataset`` from the PyTorch
core library.

For example, when accessed with ``mapdatapipe[idx]``, could read the ``idx``-th image and its
corresponding label from a folder on the disk.

.. autoclass:: MapDataPipe


By design, there are fewer ``MapDataPipe`` than ``IterDataPipe`` to avoid duplicate implementations of the same
functionalities as ``MapDataPipe``. We encourage users to use the built-in ``IterDataPipe`` for various functionalities,
and convert it to ``MapDataPipe`` as needed using ``MapToIterConverter`` or ``.to_iter_datapipe()``.
If you have any question or feedback about this design choice, feel free to open or comment on an exising Github issue
and we will be happy to discuss your use case.

We can open to add additional `MapDataPipe`s where the operations can be lazily executed and ``__len__`` can be
known in advance. Feel free to make suggestions in
`this Github issue <https://github.com/pytorch/pytorch/issues/57031>`_.

Here is the list of available Map-style DataPipes:

MapDataPipes
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
