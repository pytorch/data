# DataLoader2 (Prototype)

`DataLoader2` is introduced to execute `DataPipe` graph, with support for

- dynamic sharding for multi-process and distributed data loading
- multiple backend `ReadingServices` (WIP for `DistributedReadingService`)
- `DataPipe` graph in-place modification like shuffle control, memory pinning, etc.
- snapshot the state of data-preprocessing pipeline (WIP)

## Why DataLoader2

`DataPipe` is introduced to not only decompose the data pre-processing operations, but also decouple the overloaded
data-manipulation features from `DataLoader`. So, a light-weight `DataLoader2` is required. Besides, a certain features
can only be achieved with `DataLoader2` like snapshotting and switching backend services to perform high-performant
operations.

These options are configured by the constructor arguments of `DataLoader2`, which has the signature:

```py
DataLoader2(datapipe, datapipe_adapter_fn=None, reading_service=None)
```

The sections below describe in details the effects and usages of these options.

## DataPipe

`DataLoader2` supports both `MapDataPipe` and `IterDataPipe`. To understand the basic structure of `DataPipe`, please
see [What are DataPipes?](https://github.com/pytorch/data#what-are-datapipes).

Note: `DataLoader2` doesn't support `Dataset` or `IterableDataset`. Please wrap each of them with the corresponding
`DataPipe`.

- [`SequenceWrapper`](https://pytorch.org/data/main/generated/torchdata.datapipes.map.SequenceWrapper.html#sequencewrapper):
  `Dataset`
- [`IterableWrapper`](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.IterableWrapper.html#iterablewrapper):
  `IterableDataset`

## Data-Processing Options (`datapipe_adapter_fn`)

`Adapters` are used to configure, modify and extend the graph of `DataPipes` in `DataLoader2`. It allows in-place
modification on the pre-assembled graph of `DataPipes` provided by PyTorch domains. For example, `Shuffle(False)` can be
provided to `DataLoader2`, which would disable any of `shuffle` operations in the graph of `DataPipes`.

And, we will provide more `Adapters` to cover data-processing options (WIP):

- `PinMemory`: Attach a `DataPipe` at the end of the data-processing graph that coverts output data to `torch.Tensor` in
  pinned memory.
- `FullSync`: Attach a `DataPipe` to make sure the data-processing graph synchronized between distributed processes to
  prevent hanging.
- `ShardingPolicy`: Modify sharding policy if `sharding_filter` is presented in the graph of `DataPipes`.
- `PrefetchPolicy`, `InvalidateCache`, etc.

If you have feature requests about the `Adapters` you'd like to be provided, please open a GitHub issue. For specific
needs, `DataLoader2` also accepts any custom `Adapter` as long as it inherits from the `Adapter` class.

## ReadingService

`ReadingService` specifies the execution backend for the data-processing graph. There are two types of `ReadingServices`
in TorchData:

- `MultiprocessingReadingService`
- `DistributedReadingService`

These two `ReadingServices` would take the graph of `DataPipes` and modify it to achieve a few features like dynamic
sharding, sharing seeds and snapshoting for multi-/distributed processes.

This also allows easier transition of data-preprocessing pipeline from research to production. After the graph of
`DataPipes` is created and validated with the `ReadingServices`, a different `ReadingService` that configures and
connects to the production service/infra such as `AIStore` can be provided to `DataLoader2` as a drop-in replacement.
This `ReadingService` could potentially search the graph, and find `DataPipe` operations that can be delegated to the
production service/infra, then modify the graph correspondingly to achieve higher-performant execution.

A sequence of graph utilities are provided to help users to define their own `ReadingService` and modify the graph:

- `torchdata.dataloader2.graph.traverse(datapipe)`: Get graph of `DataPipes`
- `torchdata.dataloader2.graph.find_dps(graph, datapipe_type)`: Find `DataPipe` based on the type from the `DataPipe`
  graph
- `torchdata.dataloader2.graph.replace_dp(graph, old_datapipe, new_datapipe)`: Replace a `DataPipe` by another
  `DataPipe` in the `DataPipe` graph
- `torchdata.dataloader2.graph.remove_dp(graph, datapipe)`: Remove a `DataPipe` from the `DataPipe` graph

## Prototype Usage and Feedback

`DataLoader2` is stable in terms of API, but functionally not complete yet. We welcome early adopters and feedback, as
well as potential contributors. If you are interested in trying it out, we encourage you to install the nightly version
of this library.
