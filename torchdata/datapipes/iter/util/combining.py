# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
from collections import OrderedDict

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("zip_by_key")
class KeyZipperIterDataPipe(IterDataPipe):
    r""":class:`KeyZipperIterDataPipe`.

    Iterable datapipe to zip two datapipes based on the matching key.
    args:
        source_datapipe: KeyZipper will yield data based on the order of this datapipe
        ref_datapipe: Reference datapipe to find matching key for `source_datapipe`
        key_fn: Callable to extract key of data from source_datapipe
        ref_key_fn: Callable to extract key of data from ref_datapipe. If it's not specified, i will use same Callable as `key_fn`
        keep_key: Option to yield matching key
        buffer_size: The size of buffer used to hold key-data pair from reference datapipe. If it's specified as None, the buffer size becomes infinite
    """
    def __init__(
        self,
        source_datapipe,
        ref_datapipe,
        key_fn,
        ref_key_fn=None,
        keep_key=False,
        buffer_size=10000,
    ):
        self.source_datapipe = source_datapipe
        self.ref_datapipe = ref_datapipe
        self.key_fn = key_fn
        self.ref_key_fn = key_fn if ref_key_fn is None else ref_key_fn
        self.keep_key = keep_key
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be either None or positive integer.")
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = OrderedDict()
        ref_it = iter(self.ref_datapipe)
        warn_once_flag = True
        for data in self.source_datapipe:
            key = self.key_fn(data)
            while key not in buffer:
                try:
                    ref_data = next(ref_it)
                except StopIteration:
                    raise BufferError("No matching key can be found from reference DataPipe for the data {}. Please consider increase buffer size.".format(data))
                ref_key = self.ref_key_fn(ref_data)
                if ref_key in buffer:
                    raise ValueError("Duplicate key is found in reference DataPipe")
                if self.buffer_size is not None and len(buffer) > self.buffer_size:
                    if warn_once_flag:
                        warn_once_flag = False
                        warnings.warn("Buffer reaches the upper limit, so reference key-data pair begins to "
                                      "be removed from buffer in FIFO order. Please consider increase buffer size.")
                    buffer.popitem(last=False)
                buffer[ref_key] = ref_data
            if self.keep_key:
                yield key, data, buffer.pop(key)
            else:
                yield data, buffer.pop(key)

    def __len__(self):
        return len(self.source_datapipe)
