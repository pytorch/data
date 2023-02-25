:tocdepth: 3

.. currentmodule:: torchdata.datapipes.iter

ReadingService
===============

``ReadingService`` handles in-place modification of ``DataPipe`` graph based on different use cases.

Features
---------

Dynamic Sharding
^^^^^^^^^^^^^^^^

Dynamic sharding is achieved by ``MultiProcessingReadingService`` and ``DistributedReadingService`` to shard the pipeline based on the information of corresponding multiprocessing and distributed workers. And, TorchData offers two types of ``DataPipe`` letting users to define the sharding place within the pipeline.

- ``sharding_filter`` (:class:`ShardingFilter`): When the pipeline is replicable, each distributed/multiprocessing worker loads data from its own replica of the ``DataPipe`` graph, while skipping samples that do not belong to the corresponding worker at the point where ``sharding_filter`` is placed.

- ``sharding_round_robin_dispatch`` (:class:`ShardingRoundRobinDispatcher`): When there is any ``sharding_round_robin_dispatch`` ``DataPipe`` in the pipeline, that branch (i.e. all DataPipes prior to ``sharding_round_robin_dispatch``) will be treated as a non-replicable branch. A single dispatching process will be created to load data from the non-replicable branch and distributed data to the subsequent worker processes.

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

.. module:: torchdata.dataloader2

Determinism
^^^^^^^^^^^^

In ``DataLoader2``, a ``SeedGenerator`` becomes a single source of randomness and each ``ReadingService`` would access to it via ``initialize_iteration()`` and generate corresponding random seeds for random ``DataPipe`` operations.

In order to make sure that the Dataset shards are mutually exclusive and collectively exhaustive on multiprocessing processes and distributed nodes, ``MultiProcessingReadingService`` and ``DistributedReadingService`` would help :class:`DataLoader2` to synchronize random states for any random ``DataPipe`` operation prior to ``sharding_filter`` or ``sharding_round_robin_dispatch``. For the remaining ``DataPipe`` operations after sharding, unique random states are generated based on the distributed rank and worker process id by each ``ReadingService``, in order to perform different random transformations.

Graph Mode
^^^^^^^^^^^

This also allows easier transition of data-preprocessing pipeline from research to production. After the ``DataPipe`` graph is created and validated with the ``ReadingServices``, a different ``ReadingService`` that configures and connects to the production service/infra such as ``AIStore`` can be provided to :class:`DataLoader2` as a drop-in replacement. The ``ReadingService`` could potentially search the graph, and find ``DataPipe`` operations that can be delegated to the production service/infra, then modify the graph correspondingly to achieve higher-performant execution.

Extend ReadingService
----------------------

The followings are interfaces for custom ``ReadingService``.

.. autoclass:: ReadingServiceInterface
    :members:

The checkpoint/snapshotting feature is a work in progress. Here is the preliminary interface (small changes are likely):

.. autoclass:: CheckpointableReadingServiceInterface
    :members:

Graph Functions
^^^^^^^^^^^^^^^^
And, graph utility functions are provided in ``torchdata.dataloader.graph`` to help users to do ``DataPipe`` graph rewrite for custom ``ReadingService``:

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
