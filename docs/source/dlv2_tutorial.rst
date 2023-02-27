DataLoader2 Tutorial
=====================

This is the tutorial for users to create a ``DataPipe`` graph and load data via ``DataLoader2`` with different backend systems (``ReadingService``). An usage example can be found in `this colab notebook <https://colab.research.google.com/drive/1eSvp-eUDYPj0Sd0X_Mv9s9VkE8RNDg1u>`_.

DataPipe
---------

Please refer to `DataPipe Tutorial <dp_tutorial.html>`_ for more details. Here are the most important caveats necessary:
to make sure the data pipeline has different order per epoch and data shards are mutually exclusive and collectively exhaustive:

- Place ``sharding_filter`` or ``sharding_round_robin_dispatch`` as early as possible in the pipeline to avoid repeating expensive operations in worker/distributed processes.
- Add a ``shuffle`` DataPipe before sharding to achieve inter-shard shuffling. ``ReadingService`` will handle synchronization of those ``shuffle`` operations to ensure the order of data are the same before sharding so that all shards are mutually exclusive and collectively exhaustive.

Here is an example of a ``DataPipe`` graph:

.. code:: python

    datapipe = IterableWrapper(["./train1.csv", "./train2.csv"])
    datapipe = datapipe.open_files(encoding="utf-8").parse_csv()
    datapipe = datapipe.shuffle().sharding_filter()
    datapipe = datapipe.map(fn).batch(8)

Multiprocessing
----------------

``MultiProcessingReadingService`` handles multiprocessing sharding at the point of ``sharding_filter`` and synchronizes the seeds across worker processes.

.. code:: python

    rs = MultiProcessingReadingService(num_workers=4)
    dl = DataLoader2(datapipe, reading_service=rs)
    for epoch in range(10):
        dl.seed(epoch)
        for d in dl:
            model(d)
    dl.shutdown()

Distributed
------------

``DistributedReadingService`` handles distributed sharding at the point of ``sharding_filter`` and synchronizes the seeds across distributed processes. And, in order to balance the data shards across distributed nodes, a ``fullsync`` ``DataPipe`` will be attached to the ``DataPipe`` graph to align the number of batches across distributed ranks. This would prevent hanging issue caused by uneven shards in distributed training.

.. code:: python

    rs = DistributedReadingService()
    dl = DataLoader2(datapipe, reading_service=rs)
    for epoch in range(10):
        dl.seed(epoch)
        for d in dl:
            model(d)
    dl.shutdown()

Multiprocessing + Distributed
------------------------------

``SequentialReadingService`` can be used to combine both ``ReadingServices`` together to achieve multiprocessing and distributed training at the same time.

.. code:: python

    mp_rs = MultiProcessingReadingService(num_workers=4)
    dist_rs = DistributedReadingService()
    rs = SequentialReadingService(dist_rs, mp_rs)

    dl = DataLoader2(datapipe, reading_service=rs)
    for epoch in range(10):
        dl.seed(epoch)
        for d in dl:
            model(d)
    dl.shutdown()
