from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.batch import Batcher
from torchdata.nodes.filter import Filter
from torchdata.nodes.loader import Loader
from torchdata.nodes.prefetch import Prefetcher
from torchdata.nodes.samplers.stop_criteria import StopCriteria

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

node = IterableWrapper(a)


def is_even(x):
    return x % 2 == 0


filtered_node = Filter(node, is_even, num_workers=2)
for item in filtered_node:
    print(item)
