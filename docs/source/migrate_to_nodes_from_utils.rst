.. _migrate-to-nodes-from-utils:

Migrating to ``torchdata.nodes`` from ``torch.utils.data``
==========================================================

This guide is intended to help people familiar with ``torch.utils.data``, or
:class:`~torchdata.stateful_dataloader.StatefulDataLoader`,
to get started with ``torchdata.nodes``, and provide a starting ground for defining
your own dataloading pipelines.

We'll demonstrate how to achieve the most common DataLoader features, re-use existing samplers and datasets,
and load/save dataloader state.

Map-Style Datasets
~~~~~~~~~~~~~~~~~~

Let's look at the ``DataLoader`` constructor args and go from there

.. code:: python

    class DataLoader:
        def __init__(
            self,
            dataset: Dataset[_T_co],
            batch_size: Optional[int] = 1,
            shuffle: Optional[bool] = None,
            sampler: Union[Sampler, Iterable, None] = None,
            batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
            num_workers: int = 0,
            collate_fn: Optional[_collate_fn_t] = None,
            pin_memory: bool = False,
            drop_last: bool = False,
            timeout: float = 0,
            worker_init_fn: Optional[_worker_init_fn_t] = None,
            multiprocessing_context=None,
            generator=None,
            *,
            prefetch_factor: Optional[int] = None,
            persistent_workers: bool = False,
            pin_memory_device: str = "",
            in_order: bool = True,
        ):
            ...

As a referesher, here is roughly how dataloading works in ``torch.utils.data.DataLoader``:
``DataLoader`` begins by generating indices from a ``sampler`` and creates batches of `batch_size` indices.
If no sampler is provided, then a RandomSampler or SequentialSampler is created by default.
The indices are passed to ``Dataset.__getitem__()``, and then a ``collate_fn`` is applied to the batch
of samples. If ``num_workers > 0``, it will use multi-processing to create
subprocesses, and pass the batches of indices to the worker processes, who will then call ``Dataset.__getitem__()`` and apply ``collate_fn``
before returning the batches to the main process. At that point, ``pin_memory`` may be applied to the tensors in the batch.

Now let's look at what an equivalent implementation for DataLoader might look like, built with ``torchdata.nodes``.

.. code:: python

    from typing import List, Callable
    import torchdata.nodes as tn
    from torch.utils.data import RandomSampler, SequentialSampler, default_collate, Dataset

    class MapAndCollate:
        """A simple transform that takes a batch of indices, maps with dataset, and then applies collate
        TODO: make this a standard utility in torchdata.nodes
        """
        def __init__(self, dataset, collate_fn):
            self.dataset = dataset
            self.collate_fn = collate_fn

        def __call__(self, batch_of_indices: List[int]):
            batch = [self.dataset[i] for i in batch_of_indices]
            return self.collate_fn(batch)

    # To keep things simple, let's assume that the following args are provided by the caller
    def NodesDataLoader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        collate_fn: Callable | None,
        pin_memory: bool,
        drop_last: bool,
    ):
        # Assume we're working with a map-style dataset
        assert hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")
        # Start with a sampler, since caller did not provide one
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        # Sampler wrapper converts a Sampler to a BaseNode
        node = tn.SamplerWrapper(sampler)

        # Now let's batch sampler indices together
        node = tn.Batcher(node, batch_size=batch_size, drop_last=drop_last)

        # Create a Map Function that accepts a list of indices, applies getitem to it, and then collates them
        map_and_collate = MapAndCollate(dataset, collate_fn or default_collate)

        # MapAndCollate is doing most of the heavy lifting, so let's parallelize it. We could choose process or thread workers.
        # Note that if you're not using Free-Threaded Python (eg 3.13t) with -Xgil=0, then multi-threading might result
        # in GIL contention, and slow down training.
        node = tn.ParallelMapper(
            node,
            map_fn=map_and_collate,
            num_workers=num_workers,
            method="process",
            in_order=True,
        )

        # Optionally apply pin-memory, and we usually do some pre-fetching
        if pin_memory:
            node = tn.PinMemory(node)
        node = tn.Prefetcher(node, prefetch_factor=num_workers * 2)

        # Note that node is an iterator, and once it's exhausted, you'll need to call .reset() on it
        # to start a new Epoch.
        # We wrap the node in a Loader, which is an iterable and handles reset. It also provides
        # state_dict and load_state_dict methods
        return tn.Loader(node)


    # Now let's create a silly dataset
    class SquaredDataset(Dataset):
        def __init__(self, len: int):
            self.len = len
        def __len__(self):
            return self.len
        def __getitem__(self, i: int) -> int:
            return i**2


    # Here's how to use the state features
    loader = NodesDataLoader(
        dataset=SquaredDataset(14),
        batch_size=3,
        shuffle=False,
        num_workers=2,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
    )

    batches = []
    for idx, batch in enumerate(loader):
        if idx == 2:
            state_dict = loader.state_dict()
            # Saves the state_dict after batch 2 has been returned
        batches.append(batch)

    loader.load_state_dict(state_dict)
    batches_after_loading = list(loader)

    # Let's also compare this to torch.utils.data.DataLoader
    loaderv1 = torch.utils.data.DataLoader(
        dataset=SquaredDataset(14),
        batch_size=3,
        shuffle=False,
        num_workers=2,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,  # Coming soon to torchdata.nodes!
    )

    print("torch.utils.data.DataLoader:\n", list(loaderv1))
    print("Full dataset:\n", batches)
    print("After loading state_dict:", batches_after_loading)
    # torch.utils.data.DataLoader:
    # [tensor([0, 1, 4]), tensor([ 9, 16, 25]), tensor([36, 49, 64]), tensor([ 81, 100, 121]), tensor([144, 169])]
    # Full dataset:
    # [tensor([0, 1, 4]), tensor([ 9, 16, 25]), tensor([36, 49, 64]), tensor([ 81, 100, 121]), tensor([144, 169])]
    # After loading state_dict: [tensor([ 81, 100, 121]), tensor([144, 169])]

IterableDatasets
~~~~~~~~~~~~~~~~

Coming soon! While you can already plug your IterableDataset into an ``tn.IterableWrapper``, some functions like
``get_worker_info`` are not currently supported yet. However we believe that often, sharding work between
multi-process workers is not actually necessary, and you can keep some sort of indexing in the main process while
only parallelizing some of the heavier transforms, similar to how Map-style Datasets work above.
