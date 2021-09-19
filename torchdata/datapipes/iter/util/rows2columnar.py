# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from typing import List

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("rows2columnar")
class Rows2ColumnarIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, column_names: List[str] = [""]):
        self.source_datapipe = source_datapipe
        self.column_names = column_names

    def __iter__(self):
        for batch in self.source_datapipe:
            columnar = defaultdict(list)
            for list_or_dict_row in batch:
                if isinstance(list_or_dict_row, dict):
                    # if column_names provided, we use it as a filter
                    if len(self.column_names) > 0:
                        for column_name in self.column_names:
                            # this line will raise a KeyError if column_name
                            # is not within list_or_dict_row which is the
                            # expected bahavior
                            columnar[column_name].append(
                                list_or_dict_row[column_name])
                    else:
                        for k, v in list_or_dict_row.items():
                            columnar[k].append(v)
                else:
                    for i, v in enumerate(list_or_dict_row):
                        columnar[self.column_names[i]].append(v)
            yield columnar

    def __len__(self):
        return len(self.source_datapipe)
