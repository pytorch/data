What is ``torchdata.nodes`` (beta)?
===================================

``torchdata.nodes`` is a library of composable iterators (not
iterables!) that let you chain together common dataloading and pre-proc
operations. It follows a streaming programming model, although “sampler
+ Map-style” can still be configured if you desire.

``torchdata.nodes`` adds more flexibility to the standard
``torch.utils.data`` offering, and introduces multi-threaded parallelism
in addition to multi-process (the only supported approach in
``torch.utils.data.DataLoader``), as well as first-class support for
mid-epoch checkpointing through a ``state_dict/load_state_dict``
interface.

``torchdata.nodes`` strives to include as many useful operators as
possible, however it’s designed to be extensible. New nodes are required
to subclass ``torchdata.nodes.BaseNode``, (which itself subclasses
``typing.Iterator``) and implement ``next()``, ``reset(initial_state)``
and ``get_state()`` operations (notably, not ``__next__``,
``load_state_dict``, nor ``state_dict``)

See :doc:`getting_started_with_torchdata_nodes` to get started

Why ``torchdata.nodes``?
----------------------------------------

We get it, ``torch.utils.data`` just works for many many use cases.
However it definitely has a bunch of rough spots:

Multiprocessing sucks
~~~~~~~~~~~~~~~~~~~~~

-  You need to duplicate memory stored in your Dataset (because of
   Python copy-on-read)
-  IPC is slow over multiprocess queues and can introduce slow startup
   times
-  You’re forced to perform batching on the workers instead of
   main-process to reduce IPC overhead, increasing peak memory.
-  With GIL-releasing functions and Free-Threaded Python,
   multi-threading may not be GIL-bound like it used to be.

``torchdata.nodes`` enables both multi-threading and multi-processing so
you can choose what works best for your particular set up. Parallelism
is primarily configured in Mapper operators giving you flexibility in
the what, when, and how to parallelize.

Map-style and random-access doesn’t scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current map dataset approach is great for datasets that fit in memory,
but true random-access is not going to be very performant once your
dataset grows beyond memory limitations unless you jump through some
hoops with a special sampler.

``torchdata.nodes`` follows a streaming data model, where operators are
Iterators that can be combined together to define a dataloading and
pre-proc pipeline. Samplers are still supported (see :ref:`migrate-to-nodes-from-utils`) and
can be combined with a Mapper to produce an Iterator

Multi-Datasets do not fit well with the current implementation in ``torch.utils.data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current Sampler (one per dataloader) concepts start to break down
when you start trying to combine multiple datasets. (For single
datasets, they’re a great abstraction and will continue to be
supported!)

-  For multi-datasets, consider this scenario: ``len(dsA): 10``
   ``len(dsB): 20``. Now we want to do round-robin (or sample uniformly)
   between these two datasets to feed to our trainer. With just a single
   sampler, how can you implement that strategy? Maybe a sampler that
   emits tuples? What if you want to swap with RandomSampler, or
   DistributedSampler? How will ``sampler.set_epoch`` work?

``torchdata.nodes`` helps to address and scale multi-dataset dataloading
by only dealing with Iterators, thereby forcing samplers and datasets
together, focusing on composing smaller primitives nodes into a more
complex dataloading pipeline.

IterableDataset + multiprocessing requires additional dataset sharding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset sharding is required for data-parallel training, which is fairly
reasonable. But what about sharding between dataloader workers? With
Map-style datasets, distribution of work between workers is handled by
the main process, which distributes sampler indices to workers. With
IterableDatasets, each worker needs to figure out (through
``torch.utils.data.get_worker_info``) what data it should be returning.

.. _how-does-nodes-perform:

How does ``torchdata.nodes`` perform?
-------------------------------------

We presented some results from an early version of ``torchdata.nodes``
on a video-decoding benchmark at `PyTorch Conf 2024 <https://pytorch2024.sched.com/event/1fHn5/blobs-to-clips-efficient-end-to-end-video-data-loading-andrew-ho-ahmad-sharif-meta>`_
where we showed that:

* torchdata.nodes performs on-par or better with ``torch.utils.data.DataLoader``
  when using multi-processing (see :ref:`migrate-to-nodes-from-utils`)

* With GIL python, torchdata.nodes with multi-threading performs better than
  multi-processing in some scenarios, but makes features like GPU pre-proc
  easier to perform, which can boost throughput for many use cases.

* With No-GIL / Free-Threaded python (3.13t), we ran a benchmark loading the
  Imagenet dataset from disk, and manage to saturate main-memory bandwidth
  at a significantly lower CPU utilization than with multi-process workers
  (blogpost expected eary 2025).  See
  `imagenet_benchmark.py <https://github.com/pytorch/data/blob/main/examples/nodes/imagenet_benchmark.py>`_
  to try on your own hardware.


Design choices
--------------

No Generator BaseNodes
~~~~~~~~~~~~~~~~~~~~~~

See https://github.com/pytorch/data/pull/1362 for more thoughts.

One difficult choice we made was to disallow Generators when defining a
new BaseNode implementation. However we dropped it and moved to an
Iterator-only foundation for a few reasons around state management:

1. We require explicit state handling in BaseNode implementations.
   Generators store state implicitly on the stack and we found that we
   needed to jump through hoops and write very convoluted code to get
   basic state working with Generators
2. End-of-iteration state dict: Iterables may feel more natural, however
   a bunch of issues come up around state management. Consider the
   end-of-iteration state dict. If you load this state_dict into your
   iterable, should this represent the end-of-iteration or the start of
   the next iteration?
3. Loading state: If you call load_state_dict() on an iterable, most
   users would expect the next iterator requested from it to start with
   the loaded state. However what if iter is called twice before
   iteration begins?
4. Multiple Live Iterator problem: if you have one instance of an
   Iterable, but two live iterators, what does it mean to call
   state_dict() on the Iterable? In dataloading, this is very rare,
   however we still need to work around it and make a bunch of
   assumptions. Forcing devs that are implementing BaseNodes to reason
   about these scenarios is, in our opinion, worse than disallowing
   generators and Iterables.

``torchdata.nodes.BaseNode`` implementations are Iterators. Iterators
define ``next()``, ``get_state()``, and ``reset(initial_state | None)``.
All re-initialization should be done in reset(), including initializing
with a particular state if one is passed.

However, end-users are used to dealing with Iterables, for example,

.. code:: python

   for epoch in range(5):
     # Most frameworks and users don't expect to call loader.reset()
     for batch in loader:
       ...
     sd = loader.state_dict()
     # Loading sd should not throw StopIteration right away, but instead start at the next epoch

To handle this we keep all of the assumptions and special end-of-epoch
handling in a single ``Loader`` class which takes any BaseNode and makes
it an Iterable, handling the reset() calls and end-of-epoch state_dict
loading.
