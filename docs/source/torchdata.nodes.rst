:tocdepth: 3

torchdata.nodes
===================

torchdata.nodes is a library of composable iterators (not iterables!) that let you chain together common dataloading and pre-proc operations. It follows a streaming programming model, although "sampler + Map-style" can still be configured if you desire.

torchdata.nodes adds more flexibility to the standard torch.utils.data offering, and introduces multi-threaded parallelism in addition to multi-process (the only supported approach in torch.utils.data.DataLoader), as well as first-class support for mid-epoch checkpointing through a state_dict/load_state_dict interface.

More information on torchdata.nodes can be found in the `README <https://github.com/pytorch/data/blob/main/torchdata/nodes/README.md>`_.
