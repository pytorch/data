Getting Started With ``torchdata.nodes`` (beta)
===============================================

Install torchdata with pip.

.. code:: bash

    pip install torchdata>=0.10.0

Generator Example
~~~~~~~~~~~~~~~~~

Wrap a generator (or any iterable) to convert it to a BaseNode and get started

.. code:: python

    from torchdata.nodes import IterableWrapper, ParallelMapper, Loader

    node = IterableWrapper(range(10))
    node = ParallelMapper(node, map_fn=lambda x: x**2, num_workers=3, method="thread")
    loader = Loader(node)
    result = list(loader)
    print(result)
    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

Sampler Example
~~~~~~~~~~~~~~~

Samplers are still supported, and you can use your existing
``torch.utils.data.Dataset``\'s. See :ref:`migrate-to-nodes-from-utils` for an in-depth
example.

.. code:: python

   from torch.utils.data import RandomSampler
   from torchdata.nodes import SamplerWrapper, ParallelMapper, Loader


   class SquaredDataset(torch.utils.data.Dataset):
       def __getitem__(self, i: int) -> int:
           return i**2
       def __len__(self):
           return 10

   dataset = SquaredDataset()
   sampler = RandomSampler(dataset)

   # For fine-grained control of iteration order, define your own sampler
   node = SamplerWrapper(sampler)
   # Simply apply dataset's __getitem__ as a map function to the indices generated from sampler
   node = ParallelMapper(node, map_fn=dataset.__getitem__, num_workers=3, method="thread")
   # Loader is used to convert a node (iterator) into an Iterable that may be reused for multi epochs
   loader = Loader(node)
   print(list(loader))
   # [25, 36, 9, 49, 0, 81, 4, 16, 64, 1]
   print(list(loader))
   # [0, 4, 1, 64, 49, 25, 9, 16, 81, 36]
