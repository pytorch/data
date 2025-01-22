import itertools

from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.prefetch import Prefetcher

from torchdata.nodes.samplers.multi_node_weighted_sampler import MultiNodeWeightedSampler
from torchdata.nodes.samplers.stop_criteria import StopCriteria

from utils import DummyIterableDataset, run_test_save_load_state


num_samples = 5
num_datasets = 4

datasets = {
    f"ds{i}": IterableWrapper(DummyIterableDataset(num_samples, f"ds{i}"))
    for i in range(num_datasets)
}

weights_fn = lambda i: 0.1 * (i + 1)
weights = {f"ds{i}": weights_fn(i) for i in range(num_datasets)}
mixer = MultiNodeWeightedSampler(datasets, weights, StopCriteria.FIRST_DATASET_EXHAUSTED, seed=42)

num_epochs = 1

for epoch in range(num_epochs):
    results = list(mixer)
    
    datasets_in_results = [result["name"] for result in results]

    dataset_counts_in_results = [datasets_in_results.count(f"ds{i}") for i in range(num_datasets)]
    elements = [[result["name"], result["step"]] for result in results]
    print(elements)
    print(datasets_in_results)
    print(dataset_counts_in_results)

    mixer.reset()
