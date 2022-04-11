# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        column_names: if each element in a batch contains `Dict`, ``column_names`` act as a filter for matching keys;
            otherwise, these are used as keys to for the generated `Dict` of each batch

    Example:
        >>> # Each element in a batch is a `Dict`
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper([[{'a': 1}, {'b': 2, 'a': 1}], [{'a': 1, 'b': 200}, {'b': 2, 'c': 3, 'a': 100}]])
        >>> row2col_dp = dp.rows2columnar()
        >>> list(row2col_dp)
        [defaultdict(<class 'list'>, {'a': [1, 1], 'b': [2]}),
         defaultdict(<class 'list'>, {'a': [1, 100], 'b': [200, 2], 'c': [3]})]
        >>> row2col_dp = dp.rows2columnar(column_names=['a'])
        >>> list(row2col_dp)
        [defaultdict(<class 'list'>, {'a': [1, 1]}),
         defaultdict(<class 'list'>, {'a': [1, 100]})]
        >>> # Each element in a batch is a `List`
        >>> dp = IterableWrapper([[[0, 1, 2, 3], [4, 5, 6, 7]]])
        >>> row2col_dp = dp.rows2columnar(column_names=["1st_in_batch", "2nd_in_batch", "3rd_in_batch", "4th_in_batch"])
        >>> list(row2col_dp)
        [defaultdict(<class 'list'>, {'1st_in_batch': [0, 4], '2nd_in_batch': [1, 5],
                                      '3rd_in_batch': [2, 6], '4th_in_batch': [3, 7]})]
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
