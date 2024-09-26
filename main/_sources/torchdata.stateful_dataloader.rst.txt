:tocdepth: 3

Stateful DataLoader
===================

.. automodule:: torchdata.stateful_dataloader

StatefulDataLoader is a drop-in replacement for `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ which offers ``state_dict`` / ``load_state_dict`` methods for handling mid-epoch checkpointing which operate on the previous/next iterator requested from the dataloader (resp.).

By default, the state includes the number of batches yielded and uses this to naively fast-forward the sampler (map-style) or the dataset (iterable-style). However if the sampler and/or dataset include ``state_dict`` / ``load_state_dict`` methods, then it will call them during its own ``state_dict`` / ``load_state_dict`` calls. Under the hood, :class:`StatefulDataLoader` handles aggregation and distribution of state across multiprocess workers (but not across ranks).

.. autoclass:: StatefulDataLoader
    :members:
