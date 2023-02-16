:tocdepth: 3

DataLoader2
============

.. automodule:: torchdata.dataloader2

A new, light-weight :class:`DataLoader2` is introduced to decouple the overloaded data-manipulation functionalities from ``torch.utils.data.DataLoader`` to ``DataPipe`` operations. Besides, certain features can only be achieved with :class:`DataLoader2` like snapshotting and switching backend services to perform high-performant operations.

DataLoader2
------------

.. autoclass:: DataLoader2
    :special-members: __iter__
    :members:

Note:
:class:`DataLoader2` doesn't support ``torch.utils.data.Dataset`` or ``torch.utils.data.IterableDataset``. Please wrap each of them with the corresponding ``DataPipe`` below:

- :class:`torchdata.datapipes.map.SequenceWrapper`: ``torch.utils.data.Dataset``
- :class:`torchdata.datapipes.iter.IterableWrapper`: ``torch.utils.data.IterableDataset``

ReadingService
---------------

``ReadingService`` specifies the execution backend for the data-processing graph. There are three types of ``ReadingServices`` provided in TorchData:

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class_method_template.rst

    DistributedReadingService
    MultiProcessingReadingService
    SequentialReadingService

Each ``ReadingServices`` would take the ``DataPipe`` graph and rewrite it to achieve a few features like dynamic sharding, sharing random seeds and snapshoting for multi-/distributed processes. For more detail about those features, please refer to `the documentation <reading_service.html>`_.

Adapter
--------

``Adapter`` is used to configure, modify and extend the ``DataPipe`` graph in :class:`DataLoader2`. It allows in-place
modification or replace the pre-assembled ``DataPipe`` graph provided by PyTorch domains. For example, ``Shuffle(False)`` can be
provided to :class:`DataLoader2`, which would disable any ``shuffle`` operations in the ``DataPipes`` graph.

.. module:: torchdata.dataloader2.adapter

.. autoclass:: Adapter
    :special-members: __call__

Here are the list of :class:`Adapter` provided by TorchData in ``torchdata.dataloader2.adapter``:

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class_template.rst

    Shuffle
    CacheTimeout

And, we will provide more ``Adapters`` to cover data-processing options:

- ``PinMemory``: Attach a ``DataPipe`` at the end of the data-processing graph that coverts output data to ``torch.Tensor`` in pinned memory.
- ``FullSync``: Attach a ``DataPipe`` to make sure the data-processing graph synchronized between distributed processes to prevent hanging.
- ``ShardingPolicy``: Modify sharding policy if ``sharding_filter`` is presented in the ``DataPipe`` graph.
- ``PrefetchPolicy``, ``InvalidateCache``, etc.

If you have feature requests about the ``Adapters`` you'd like to be provided, please open a GitHub issue. For specific
needs, ``DataLoader2`` also accepts any custom ``Adapter`` as long as it inherits from the ``Adapter`` class.
