#This is a local file for testing and will be deleted in the future.
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.batch import Batcher
from torchdata.nodes.filter import Filter
from torchdata.nodes.loader import Loader
from torchdata.nodes.prefetch import Prefetcher
from torchdata.nodes.samplers.stop_criteria import StopCriteria
from utils import MockSource, run_test_save_load_state, StatefulRangeNode


a = list(range(60))
base_node = IterableWrapper(a)


def is_even(x):
    return x % 2 == 0


node = Filter(base_node, is_even, num_workers=2)

print(node.get_state())
for _ in range(28):
    print(next(node))
print(node.get_state())

state = node.get_state()
node.reset()

print(node.get_state())

for _ in range(2):
    print(next(node))

del node
node = Filter(base_node, is_even, num_workers=2)
print("state to be loaded", state)
print("state before reset", node.get_state())
node.reset(state)
print(node.get_state())

for item in node:
    print(item)
