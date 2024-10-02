# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from torch.utils.data.sampler import Sampler

from .datasource_sampler import SamplerSettings
from .datasource import DataSource

class DataSourceReader():
    def __init__(self, datasource: DataSource, shuffle=False, sampler=None):
        self.datasource = datasource
        self.shuffle = shuffle
        self.sampler = sampler

        self.sampler_settings = None
        self.num_bundles = self.datasource.get_num_bundles()
        self.get_samples_per_bundle = self.datasource.get_samples_per_bundle()
        self.length = sum(self.get_samples_per_bundle)
        self.rank, self.world_size = get_rank_and_world_size()

    def set_sampler(
        self, sampler: Union[Sampler, SamplerSettings]
    ) -> None:
        self.sampler, self.sampler_settings = None, None
        if isinstance(sampler, SamplerSettings):
            self.sampler_settings = sampler
        elif isinstance(sampler, Sampler):
            self.sampler = sampler
        self.batch_sampler = None

    def get_sampler(self) -> Sampler[int]:
        # if get_sample() is avaialble with bundles
        if self.sampler is None:
            self.sampler = self._set_up_sampler(self.sampler_settings)
        if self.sampler is None:
            raise NotImplementedError("Sampler is not set up")
        return self.sampler
        
    def _set_up_sampler(self, sampler_settings: Optional[SamplerSettings]) -> Sampler[int]:
        if sampler_settings is None:
            return None

        self.rank_to_rows = __setup_rank_to_rows(sampler_settings):

        if sampler_settings.sampler_type == SamplerType.PSEUDORANDOM:
            return PseudoRandomSampler(
                num_samples=sampler_settings.num_samples,
                seed=sampler_settings.seed,
            )
        elif sampler_settings.sampler_type == SamplerType.INORDER:
            return InOrderSampler(
                num_samples=sampler_settings.num_samples,
            )
        
    def __setup_rank_to_rows(self, sampler_settings: Optional[SamplerSettings]) -> Optional[List[Tuple[int, int]]]:
        rows_per_rank = [self.length // self.world_size] * self.world_size
        # Evenly distribute the remainder of samples to the ranks if we are not dropping last
        # e.g. [3, 3, 3, 2]
        if not settings.drop_last:
            for i in range(self.length % self.world_size):
                rows_per_rank[i] += 1

        start_range = 0
        rank_to_rows = []
        for num_rows in rows_per_rank:
            rank_to_rows.append([(start_range, start_range + num_rows)])
            start_range += num_rows

        if len(rank_to_rows) != self.world_size:
            raise ValueError(
                f"rank_to_rows should have length {self.world_size=} "
                f"but got {len(rank_to_rows)=}, {rank_to_rows=}",
            )
            
        return rank_to_rows

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
    
