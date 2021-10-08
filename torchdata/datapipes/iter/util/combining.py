# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
from collections import OrderedDict

from torch.utils.data import IterDataPipe, MapDataPipe, functional_datapipe
from typing import Callable, Optional


@functional_datapipe("zip_with_iter")
class IterZipperIterDataPipe(IterDataPipe):
    r""":class:`IterZipperIterDataPipe`.

    Iterable DataPipe to zip two DataPipes based on the matching key.

    Args:
        source_datapipe: IterZipper will yield data based on the order of this DataPipe
        ref_datapipe: Reference datapipe to find matching key for `source_datapipe`
        key_fn: Callable to extract key of data from source_datapipe
        ref_key_fn: Callable to extract key of data from ref_datapipe.
            If it's not specified, the `key_fn` would be applied to reference data
        keep_key: Option to yield matching key
        buffer_size: The size of buffer used to hold key-data pair from reference DataPipe.
            If it's specified as None, the buffer size becomes infinite
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        ref_datapipe: IterDataPipe,
        key_fn: Callable,
        ref_key_fn: Optional[Callable] = None,
        keep_key: bool = False,
        buffer_size: Optional[int] = 10000,
    ):
        if not isinstance(ref_datapipe, IterDataPipe):
            raise TypeError(f"ref_datapipe must be a IterDataPipe, but its type is {type(ref_datapipe)} instead.")
        self.source_datapipe = source_datapipe
        self.ref_datapipe = ref_datapipe
        self.key_fn = key_fn
        self.ref_key_fn = key_fn if ref_key_fn is None else ref_key_fn
        self.keep_key = keep_key
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be either None or a positive integer.")
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer: OrderedDict = OrderedDict()
        ref_it = iter(self.ref_datapipe)
        warn_once_flag = True
        for data in self.source_datapipe:
            key = self.key_fn(data)
            while key not in buffer:
                try:
                    ref_data = next(ref_it)
                except StopIteration:
                    raise BufferError(
                        f"No matching key can be found from reference DataPipe for the data {data}. "
                        "Please consider increasing the buffer size."
                    )
                ref_key = self.ref_key_fn(ref_data)
                if ref_key in buffer:
                    raise ValueError("Duplicate key is found in reference DataPipe")
                if self.buffer_size is not None and len(buffer) > self.buffer_size:
                    if warn_once_flag:
                        warn_once_flag = False
                        warnings.warn(
                            "Buffer reaches the upper limit, so reference key-data pair begins to "
                            "be removed from buffer in FIFO order. Please consider increase buffer size."
                        )
                    buffer.popitem(last=False)
                buffer[ref_key] = ref_data
            if self.keep_key:
                yield key, data, buffer.pop(key)
            else:
                yield data, buffer.pop(key)

    def __len__(self):
        return len(self.source_datapipe)


def tuple_merge(item, item_from_map):
    return (item, item_from_map)


@functional_datapipe("zip_with_map")
class MapZipperIterDataPipe(IterDataPipe):
    r""" :class:`MapZipperIterDataPipe`.

    IterDataPipe that joins the items from the source IterDataPipe with items from a MapDataPipe. The
    matching is done by the key function, which maps an item from source IterDataPipe to
    a key that exists in MapDataPipe. The return value is created by the merge function, which returns
    a tuple of the two items by default.

    Args:
        source_iterdatapipe: IterDataPipe from which items are yield and will be combined with an item from map_datapipe
        map_datapipe: MapDataPipe that takes a key from key_fn, and returns an item
        key_fn: Function that maps each item from source_iterdatapipe to a key that exists in map_datapipe
        merge_fn: Function that combines the item from source_iterdatapipe and the item from map_datapipe,
            by default a tuple is created
    """

    def __init__(
        self,
        source_iterdatapipe: IterDataPipe,
        map_datapipe: MapDataPipe,
        key_fn: Callable,
        merge_fn: Callable = tuple_merge,
    ):
        if not isinstance(map_datapipe, MapDataPipe):
            raise TypeError(f"map_datapipe must be a MapDataPipe, but its type is {type(map_datapipe)} instead.")
        self.source_iterdatapipe = source_iterdatapipe
        self.map_datapipe = map_datapipe
        self.key_fn = key_fn
        self.merge_fn = merge_fn
        self.length = -1

    def __iter__(self):
        for item in self.source_iterdatapipe:
            key = self.key_fn(item)
            try:
                map_item = self.map_datapipe[key]
            except (KeyError, IndexError):
                raise KeyError(f"key_fn maps {item} to {key}, which is not a valid key in the given MapDataPipe.")
            yield self.merge_fn(item, map_item)

    def __len__(self) -> int:
        if self.length == -1:
            self.length = len(self.source_iterdatapipe)
        return self.length
