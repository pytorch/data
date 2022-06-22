# TODO: This file can be moved to the dataframe parent directory once Torcharrow
# is open sourced

from typing import List, Optional, Union, Iterable

import torcharrow as ta
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper


class TorcharrowWrapper:
    @classmethod
    def create_dataframe(cls, data: Iterable, columns: Optional[List[str]] = None):
        columnar_data = list(zip(*data))

        # set default column values if `columns` arg is not provided
        column_names = columns
        if not columns or len(columns) == 0:
            column_names = [f"col{i}" for i in range(len(columnar_data))]

        return ta.dataframe(
            {
                column_name: ta.Column(value)
                for column_name, value in zip(column_names, columnar_data)
            }
        )

    @classmethod
    def is_dataframe(cls, data: Union[ta.DataFrame, ta.Column]):
        return isinstance(data, ta.DataFrame)

    @classmethod
    def is_column(cls, data: Union[ta.DataFrame, ta.Column]):
        return isinstance(data, ta.Column)

    @classmethod
    def iterate(cls, df):
        for d in df:
            yield d

    @classmethod
    def concat(cls, buffer: List[ta.DataFrame]):
        concat_buffer = []
        for b in buffer:
            concat_buffer += list(b)
        return ta.dataframe(concat_buffer, dtype=buffer[0].dtype)

    @classmethod
    def get_item(cls, df: ta.DataFrame, idx):
        return df[idx:idx+1]

    @classmethod
    def get_len(cls, df: ta.DataFrame):
        return len(df)

    @classmethod
    def get_columns(cls, df):
        return list(df.columns)


df_wrapper.set_df_wrapper(TorcharrowWrapper)
