# Copyright (c) Facebook, Inc. and its affiliates.
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("add_index")
class IndexAdderIterDataPipe(IterDataPipe):
    r""":class:`IndexAdder`.

    Iterable DataPipe to add an index to an existing datapipe. The row or batch
    must be of type dict otherwise a `NotImplementedError` is thrown. The index
    of the data is set to the `index_name` field provided.

    Args:
        source_datapipe: Iterable DataPipe being indexed
        index_name: Name of the key to store data index
    """

    def __init__(self, source_datapipe, index_name="index") -> None:
        self.source_datapipe = source_datapipe
        self.index_name = index_name

    def __iter__(self):
        for i, row_or_batch in enumerate(self.source_datapipe):
            if isinstance(row_or_batch, dict):
                row_or_batch[self.index_name] = i
                yield row_or_batch
            else:
                raise NotImplementedError("We only support adding index to row or batch in dict type")

    def __len__(self):
        return len(self.source_datapipe)
