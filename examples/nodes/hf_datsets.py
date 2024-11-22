from datasets import Dataset  # make sure that you have huggingface datasets installed

data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.base_node import BaseNode
from torchdata.nodes.batch import Batcher
from torchdata.nodes.loader import Loader
from torchdata.nodes.map import Mapper, ParallelMapper
from torchdata.nodes.prefetch import Prefetcher

ds = Dataset.from_dict({"data": data})

source_node = IterableWrapper(ds)

print("Getting elements from the source node")
while True:
    try:
        print(next(source_node))
    except StopIteration:
        break


print("Getting elements from the source node with a batcher")
batch_size = 2
batcher = Batcher(source_node, batch_size, drop_last=True)

for batch in batcher:
    print(batch)

print("Getting elements from the source node with a batcher and drop_last=False")
batcher = Batcher(source_node, batch_size, drop_last=False)

for batch in batcher:
    print(batch)

print("Using a simple mapper function")
mapper = Mapper(source_node, lambda x: [x["data"][0] * 2, x["data"][1] ** 2])

for element in mapper:
    print(element)


# print("Using a parallel mapper function")

# mapper = ParallelMapper(
#     source_node, map_fn=lambda x: [x["data"][0] * 2, x["data"][1] ** 2], num_workers=0
# )
# for element in mapper:
#     print(element)
