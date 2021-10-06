from torch.utils.data import IterDataPipe, MapDataPipe, functional_datapipe
from typing import Callable


# TODO: Confirm if we should move this to Core or not
@functional_datapipe("hashjoin")
class HashJoinerIterDataPipe(IterDataPipe):
    r""" :class:`HashJoinerIterDataPipe`.
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
        merge_fn: Callable = lambda a, b: (a, b),
    ):
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
