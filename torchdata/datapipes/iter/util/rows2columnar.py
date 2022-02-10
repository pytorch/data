# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from typing import Dict, Iterator, List, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("rows2columnar")
class Rows2ColumnarIterDataPipe(IterDataPipe[Dict]):
    r"""
    Accepts an input DataPipe with batches of data, and processes one batch
    at a time and yields a Dict for each batch, with ``column_names`` as keys and lists of
    corresponding values from each row as values (functional name: ``rows2columnar``).

    Within the input DataPipe, each row within a batch must either be a `Dict` or a `List`

    Note:
        If ``column_names`` are not given and each row is a `Dict`, the keys of that Dict will be used as column names.

    Args:
        source_datapipe: a DataPipe where each item is a batch. Within each batch,
            there are rows and each row is a `List` or `Dict`
        column_names: a function that joins a list of lines together
    """
    column_names: List[str]

    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]], column_names: List[str] = None) -> None:
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe
        self.column_names: List[str] = [] if column_names is None else column_names

    def __iter__(self) -> Iterator[Dict]:
        for batch in self.source_datapipe:
            columnar = defaultdict(list)
            for list_or_dict_row in batch:
                if isinstance(list_or_dict_row, dict):
                    # if column_names provided, we use it as a filter
                    if len(self.column_names) > 0:
                        for column_name in self.column_names:
                            # this line will raise a KeyError if column_name
                            # is not within list_or_dict_row which is the
                            # expected behavior
                            columnar[column_name].append(list_or_dict_row[column_name])
                    else:
                        for k, v in list_or_dict_row.items():
                            columnar[k].append(v)
                else:
                    for i, v in enumerate(list_or_dict_row):
                        columnar[self.column_names[i]].append(v)
            yield columnar

    def __len__(self) -> int:
        return len(self.source_datapipe)
