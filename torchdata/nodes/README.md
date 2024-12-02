# torchdata.nodes

## What is `torchdata.nodes`?

`torchdata.nodes` is a library of composable iterators (not iterables!) that let you chain together common dataloading
and pre-proc operations. It follows a streaming programming model, although "sampler + Map-style" can still be
configured if you desire.

`torchdata.nodes` adds more flexibility to the standard `torch.utils.data` offering, and introduces multi-threaded
parallelism in addition to multi-process (the only supported approach in `torch.utils.data.DataLoader`), as well as
first-class support for mid-epoch checkpointing through a `state_dict/load_state_dict` interface.

`torchdata.nodes` strives to include as many useful operators as possible, however it's designed to be extensible. New
nodes are required to subclass `torchdata.nodes.BaseNode`, (which itself subclasses `typing.Iterator`) and implement
`next()`, `reset(initial_state)` and `get_state()` operations (notably, not `__next__`, `load_state_dict`, nor
`state_dict`)

## Getting started

Install torchdata with pip.

```bash
pip install torchdata>=0.10.0
```

### Generator Example

Wrap a generator (or any iterable) to convert it to a BaseNode and get started

```python
from torchdata.nodes import IterableWrapper, ParallelMapper, Loader

node = IterableWrapper(range(10))
node = ParallelMapper(node, map_fn=lambda x: x**2, num_workers=3, method="thread")
loader = Loader(node)
result = list(loader)
print(result)
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### Sampler Example

Samplers are still supported, and you can use your existing `torch.utils.data.Dataset`s

```python
import torch.utils.data
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
```

## What's the point of `torchdata.nodes`?

We get it, `torch.utils.data` just works for many many use cases. However it definitely has a bunch of rough spots:

### Multiprocessing sucks

- You need to duplicate memory stored in your Dataset (because of Python copy-on-read)
- IPC is slow over multiprocess queues and can introduce slow startup times
- You're forced to perform batching on the workers instead of main-process to reduce IPC overhead, increasing peak
  memory.
- With GIL-releasing functions and Free-Threaded Python, multi-threading may not be GIL-bound like it used to be.

`torchdata.nodes` enables both multi-threading and multi-processing so you can choose what works best for your
particular set up. Parallelism is primarily configured in Mapper operators giving you flexibility in the what, when, and
how to parallelize.

### Map-style and random-access doesn't scale

Current map dataset approach is great for datasets that fit in memory, but true random-access is not going to be very
performant once your dataset grows beyond memory limitations unless you jump through some hoops with a special sampler.

`torchdata.nodes` follows a streaming data model, where operators are Iterators that can be combined together to define
a dataloading and pre-proc pipeline. Samplers are still supported (see example above) and can be combined with a Mapper
to produce an Iterator

### Multi-Datasets do not fit well with the current implementation in `torch.utils.data`

The current Sampler (one per dataloader) concepts start to break down when you start trying to combine multiple
datasets. (For single datasets, they're a great abstraction and will continue to be supported!)

- For multi-datasets, consider this scenario: `len(dsA): 10` `len(dsB): 20`. Now we want to do round-robin (or sample
  uniformly) between these two datasets to feed to our trainer. With just a single sampler, how can you implement that
  strategy? Maybe a sampler that emits tuples? What if you want to swap with RandomSampler, or DistributedSampler? How
  will `sampler.set_epoch` work?

`torchdata.nodes` helps to address and scale multi-dataset dataloading by only dealing with Iterators, thereby forcing
samplers and datasets together, focusing on composing smaller primitives nodes into a more complex dataloading pipeline.
