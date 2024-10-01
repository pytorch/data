# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
from torch.utils.data import IterableDataset, Dataset, get_worker_info

from .datasource import DataSource

class DataSourceReader():
    def __init__(self, datasource: DataSource, shuffle=False, sampler=None):
        raise NotImplementedError("DataSourceReader is not implemented")

    def get_sampler(self):
        raise NotImplementedError("get_sampler is not implemented")
    
    def to_map_dataset(self):
        raise DataSourceMapDataset(self.datasource, self.shuffle)

    def to_iterable_dataset(self):
        return DataSourceIterableDataset(self.datasource, self.shuffle)

class DataSourceIterableDataset(IterableDataset):
    def __init__(self, datasource: DataSource, shuffle: bool):
        self.shuffle = shuffle
        self.datasource = datasource

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers_per_rank = worker_info.num_workers if worker_info else 1
        
        rank, world_size = get_rank_and_world_size() # distributed setup
        global_worker_rank = rank * num_workers_per_rank + worker_id
        num_global_workers = num_workers_per_rank * world_size

        epoch = 0
        # TODO: StopIteration
        while True:
            # Shuffle the bundles for each epoch
            bundle_list = shuffle(
                list(range(self.datasource.get_num_bundles()), 
                # each worker should not have a different seed per epoch
                seed=torch.initial_seed().seed+epoch
            )
            # Shard the bundles across workers
            for bundle_id in bundle_list[global_worker_rank::num_global_workers]:
                bundle = self.datasource.get_bundle(bundle_id)
                yield from bundle
            epoch += 1

class DataSourceMapDataset(Dataset):
    def __init__(self, datasource: DataSource, shuffle: bool):
        self.shuffle = shuffle
        self.datasource = datasource
        self.rows_per_bundle = self.datasource.get_samples_per_bundle()
    
    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__ is not implemented")

    def __len__(self):
        raise NotImplementedError("__len__ is not implemented")
    
