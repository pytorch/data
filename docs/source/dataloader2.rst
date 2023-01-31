:tocdepth: -1

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

``ReadingService`` specifies the execution backend for the data-processing graph. There are three types of ``ReadingServices`` in TorchData:

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class_method_template.rst

    DistributedReadingService
    MultiProcessingReadingService
    PrototypeMultiProcessingReadingService

Each ``ReadingServices`` would take the ``DataPipe`` graph and modify it to achieve a few features like dynamic sharding, sharing random seeds and snapshoting for multi-/distributed processes.

Dynamic Sharding
^^^^^^^^^^^^^^^^

Dynamic sharding is achieved by ``PrototypeMultiProcessingReadingService`` and ``DistributedReadingService`` to shard the pipeline based on the information of corresponding multiprocessing and distributed workers. And, TorchData offers two types of ``DataPipe`` letting users to define the sharding place within the pipeline.

- ``sharding_filter``: When the pipeline is replicable, each distributed/multiprocessing worker loads data from one replica of the ``DataPipe`` graph, and skip the data not blonged to the corresponding worker at the place of ``sharding_filter``.

- ``sharding_round_robin_dispatch``: When there is any ``sharding_round_robin_dispatch`` ``DataPipe`` in the pipeline, that branch will be treated as a non-replicable branch. Then, a single dispatching process will be created to load data from the non-repliable branch and distributed data to the subsequent worker processes.

The following is an example of having two types of sharding strategies in the pipeline.

.. graphviz::

    digraph Example {
        subgraph cluster_replicable {
            label="Replicable"
            a -> b -> c -> d -> l;
            color=blue;
        }

        subgraph cluster_non_replicable {
            style=filled;
            color=lightgrey;
            node [style=filled,color=white];
            label="Non-Replicable"
            e -> f -> g -> k;
            h -> i -> j -> k;
        }

        k -> l -> fullsync -> end;

        a [label="DP1"];
        b [label="shuffle"];
        c [label="sharding_filter", color=blue];
        d [label="DP4"];
        e [label="DP2"];
        f [label="shuffle"];
        g [label="sharding_round_robin_dispatch", style="filled,rounded", color=red, fillcolor=white];
        h [label="DP3"];
        i [label="shuffle"];
        j [label="sharding_round_robin_dispatch", style="filled,rounded", color=red, fillcolor=white];
        k [label="DP5 (Lowest common ancestor)"];
        l [label="DP6"];
        fullsync;
        end [shape=box];
    }

When multiprocessing takes place, the graph becomes:

.. graphviz::

    digraph Example {
        subgraph cluster_worker_0 {
            label="Worker 0"
            a0 -> b0 -> c0 -> d0 -> l0;
            m0 -> l0;
            color=blue;
        }

        subgraph cluster_worker_1 {
            label="Worker 1"
            a1 -> b1 -> c1 -> d1 -> l1;
            m1 -> l1;
            color=blue;
        }

        subgraph cluster_non_replicable {
            style=filled;
            color=lightgrey;
            node [style=filled,color=white];
            label="Non-Replicable"
            e -> f -> g -> k;
            h -> i -> j -> k;
            k -> round_robin_demux;
        }

        round_robin_demux -> m0;
        round_robin_demux -> m1;
        l0 -> n;
        l1 -> n;
        n -> fullsync -> end;

        a0 [label="DP1"];
        b0 [label="shuffle"];
        c0 [label="sharding_filter", color=blue];
        d0 [label="DP4"];
        a1 [label="DP1"];
        b1 [label="shuffle"];
        c1 [label="sharding_filter", color=blue];
        d1 [label="DP4"];
        e [label="DP2"];
        f [label="shuffle"];
        g [label="sharding_round_robin_dispatch", style="filled,rounded", color=red, fillcolor=white];
        h [label="DP3"];
        i [label="shuffle"];
        j [label="sharding_round_robin_dispatch", style="filled,rounded", color=red, fillcolor=white];
        k [label="DP5 (Lowest common ancestor)"];
        fullsync;
        l0 [label="DP6"];
        l1 [label="DP6"];
        m0 [label="Client"]
        m1 [label="Client"]
        n [label="Client"]
        end [shape=box];
    }

``Client`` in the graph is a ``DataPipe`` that send request and receive response from multiprocessing queues.

Graph Mode
^^^^^^^^^^

This also allows easier transition of data-preprocessing pipeline from research to production. After the ``DataPipe`` graph is created and validated with the ``ReadingServices``, a different ``ReadingService`` that configures and connects to the production service/infra such as ``AIStore`` can be provided to :class:`DataLoader2` as a drop-in replacement. The ``ReadingService`` could potentially search the graph, and find ``DataPipe`` operations that can be delegated to the production service/infra, then modify the graph correspondingly to achieve higher-performant execution.

The followings are interfaces for custom ``ReadingService``.

.. autoclass:: ReadingServiceInterface
    :members:

The checkpoint/snapshotting feature is a work in progress. Here is the preliminary interface (small changes are likely):

.. autoclass:: CheckpointableReadingServiceInterface
    :members:

And, graph utility functions are provided in ``torchdata.dataloader.graph`` to help users to define their own ``ReadingService`` and modify the graph:

.. module:: torchdata.dataloader2.graph

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: function.rst

    traverse_dps
    find_dps
    list_dps
    remove_dp
    replace_dp

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
