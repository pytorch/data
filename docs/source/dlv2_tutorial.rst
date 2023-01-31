DataLoader2 Tutorial
=====================

This is the tutorial for users to create ``DataPipe`` graph and load data via ``DataLoader2`` with different backend systems (``ReadingService``).

DataPipe
---------

Please refer to `DataPipe Tutorial <dp_tutorial.html>_` for more details. Here are the most important caveats:
to make sure the data pipeline has different order per epoch and data shards are mutually exclusive and collectively exhaustive:

- Place ``sharding_filter`` or ``sharding_round_robin_dispatch`` as early as possibel in the pipele to avoid repeating expensive operations in worker/distributed processes.
- Add a ``shuffle`` DataPipe before sharding to achieve inter-shard shuffling. ``ReadingService`` will handle synchronization of those ``shuffle`` operations to the order of data are the same before sharding so that all shards are mutually exclusive and collectively exhaustive.

Here is an example of ``DataPipe`` graph:

.. code:: python

    datapipe = IterableWrapper(["./train1.csv", "./train2.csv"])
    datapipe = datapipe.open_files(encoding="utf-8").parse_csv()
    datapipe = datapipe.shuffle().sharding_filter()
    datapipe = datapiep.map(fn).batch(8)

Multiprocessing
----------------

``PrototypeMultiProcessingReadingService`` handles multiprocessing sharding at the point of ``sharding_filter`` and synchronize the seeds across worker processes.

.. code:: python

    rs = PrototypeMultiProcessingReadingService(num_workers=4)
    dl = DataLoader2(datapipe, reading_service=rs)
    for epoch in range(10):
        dl.seed(epoch)
        for d in dl:
            model(d)
    dl.shutdown()

Distributed
------------

``DistributedReadingService`` handles distributed sharding at the point of ``sharding_filter`` and synchronize the seeds across distributed processes. And, in order to balance the data shards across distributed nodes, a ``fullsync`` ``DataPipe`` will be attached to the ``DataPipe`` graph to align the number of batches across distributed ranks. This would prevent hanging issue caused by uneven shards in distributed training.

.. code:: python

    rs = DistributedReadingService()
    dl = DataLoader2(datapipe, reading_service=rs)
    for epoch in range(10):
        dl.seed(epoch)
        for d in dl:
            model(d)
    dl.shutdown()
