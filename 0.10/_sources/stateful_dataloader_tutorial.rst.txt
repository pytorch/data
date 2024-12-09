Stateful DataLoader Tutorial
============================

Saving and loading state
------------------------

Stateful DataLoader adds the ``load_state_dict``, ``state_dict`` methods to the ``torch.utils.data.DataLoader``. State fetch and set can be done as follows:

.. code:: python

    from torchdata.stateful_dataloader import StatefulDataLoader

    dataloader = StatefulDataLoader(dataset, num_workers=2)
    for i, batch in enumerate(dataloader):
        ...
        if i == 10:
            state_dict = dataloader.state_dict()
            break

    # Training run resumes with the previous checkpoint
    dataloader = StatefulDataLoader(dataset, num_workers=2)
    # Resume state with DataLoader
    dataloader.load_state_dict(state_dict)
    for i, batch in enumerate(dataloader):
        ...

Saving Custom State with Map-Style Datasets
-------------------------------------------

For efficient resuming of `Map-style datasets <https://pytorch.org/docs/stable/data.html#map-style-datasets>`_, you can resume iteration by defining ``state_dict`` / ``load_state_dict`` methods in your sampler. If your dataset has worker-specific state (eg RNG transform state) you can add ``state_dict`` / ``load_state_dict`` methods to your dataset.

.. code:: python

    from typing import *
    import torch
    import torch.utils.data
    from torchdata.stateful_dataloader import StatefulDataLoader

    # If you are using the default RandomSampler and BatchSampler in torch.utils.data, they are patched when you import torchdata.stateful_dataloader so that defining, a custom sampler here is unnecessary
    class MySampler(torch.utils.data.Sampler[int]):
        def __init__(self, high: int, seed: int, limit: int):
            self.seed, self.high, self.limit = seed, high, limit
            self.g = torch.Generator()
            self.g.manual_seed(self.seed)
            self.i = 0

        def __iter__(self):
            while self.i < self.limit:
            val = int(torch.randint(high=self.high, size=(1,), generator=self.g))
            self.i += 1
            yield val

        def load_state_dict(self, state_dict: Dict[str, Any]):
            self.i = state_dict["i"]
            self.g.set_state(state_dict["rng"])

        def state_dict(self) -> Dict[str, Any]:
            return {"i": self.i, "rng": self.g.get_state()}

    # Optional: save dataset random transform state
    class NoisyRange(torch.utils.data.Dataset):
        def __init__(self, high: int, mean: float, std: float):
            self.high, self.mean, self.std = high, torch.tensor([float(mean)]), float(std)

        def __len__(self):
            return self.high

        def __getitem__(self, idx: int) -> float:
            if not (0 <= idx < self.high):
            raise IndexError()
            x = torch.normal(self.mean, self.std)
            noise = x.item()
            return idx + noise

        def load_state_dict(self, state_dict):
            torch.set_rng_state(state_dict["rng"])

        def state_dict(self):
            return {"rng": torch.get_rng_state()}

    # Test both single/multiprocess dataloading
    for num_workers in [0, 2]:
        print(f"{num_workers=}")
        dl = StatefulDataLoader(NoisyRange(5, 1, 1), sampler=MySampler(5, 1, 10),
            batch_size=2, drop_last=False, num_workers=num_workers)

    batches = []
    for i, batch in enumerate(dl):
        batches.append(batch)
        if i == 2:
        sd = dl.state_dict()

    dl.load_state_dict(sd)
    batches2 = list(dl)

    print(batches[3:])
    print(batches2)

    """
    Output:
    num_workers=0
    [tensor([-0.4526,  3.7948], dtype=torch.float64), tensor([6.5494, 3.0470], dtype=torch.float64)]
    [tensor([-0.4526,  3.7948], dtype=torch.float64), tensor([6.5494, 3.0470], dtype=torch.float64)]
    num_workers=2
    [tensor([3.7412, 1.2438], dtype=torch.float64), tensor([4.4807, 4.0036], dtype=torch.float64)]
    [tensor([3.7412, 1.2438], dtype=torch.float64), tensor([4.4807, 4.0036], dtype=torch.float64)]
    """

Saving Custom State with Iterable-Style Datasets
------------------------------------------------

Tracking iteration order with `Iterable-style datasets <https://pytorch.org/docs/stable/data.html#iterable-style-datasets>`_ requires state from each worker-level instance of the dataset to be captured. You can define ``state_dict`` / ``load_state_dict`` methods on your dataset which capture worker-level state. :class:`StatefulDataLoader` will handle aggregation across workers and distribution back to the workers. Calling ``load_state_dict`` requires :class:`StatefulDataLoader`` to have same ``num_workers`` as those of the provided ``state_dict``.

.. code:: python

    from typing import *
    import torch
    import torch.utils.data
    from torchdata.stateful_dataloader import StatefulDataLoader


    class MyIterableDataset(torch.utils.data.IterableDataset):
        def __init__(self, high: int, seed: int):
            self.high, self.seed = high, seed
            self.g = torch.Generator()
            self.i = 0

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            else:
            worker_id = 0
            num_workers = 1
            self.g.manual_seed(self.seed)
            arr = torch.randperm(self.high, generator=self.g)
            arr = arr[worker_id:self.high:num_workers]
            for idx in range(self.i, len(arr)):
            self.i += 1
            yield arr[idx]
            self.i = 0

        def state_dict(self):
            return {"i": self.i}

        def load_state_dict(self, state_dict):
            self.i = state_dict["i"]

    # Test both single/multiprocess dataloading
    for num_workers in [0, 2]:
    print(f"{num_workers=}")
    dl = StatefulDataLoader(
        MyIterableDataset(12, 0), batch_size=2, drop_last=False,
        num_workers=num_workers)

    batches = []
    for i, batch in enumerate(dl):
        batches.append(batch)
        if i == 2:
        sd = dl.state_dict()

    dl.load_state_dict(sd)
    batches2 = list(dl)

    print(batches[3:])
    print(batches2)

    """
    Output:
    num_workers=0
    [tensor([ 2, 10]), tensor([3, 1]), tensor([11,  6])]
    [tensor([ 2, 10]), tensor([3, 1]), tensor([11,  6])]
    num_workers=2
    [tensor([ 4, 10]), tensor([ 3, 11]), tensor([1, 6])]
    [tensor([ 4, 10]), tensor([ 3, 11]), tensor([1, 6])]
    """
