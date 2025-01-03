# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision
from PIL import Image


# TODO: maybe infinite buffer can / is already natively supported by torchdata?
INFINITE_BUFFER_SIZE = 1_000_000_000

IMAGENET_TRAIN_LEN = 1_281_167
IMAGENET_TEST_LEN = 50_000


def _decode(path, root, category_to_int):
    category = Path(path).relative_to(root).parts[0]

    image = Image.open(path).convert("RGB")
    label = category_to_int(category)

    return image, label


def _apply_tranforms(img_and_label, transforms):
    img, label = img_and_label
    return transforms(img), label


class PreLoadedMapStyle:
    # All the data is pre-loaded and transformed in __init__, so the DataLoader should be crazy fast.
    # This is just to assess how fast a model could theoretically be trained if there was no data bottleneck at all.
    def __init__(self, dir, transform, buffer_size=100):
        dataset = torchvision.datasets.ImageFolder(dir, transform=transform)
        self.size = len(dataset)
        self.samples = [dataset[torch.randint(0, len(dataset), size=(1,)).item()] for i in range(buffer_size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx % len(self.samples)]


class MapStyleToIterable(torch.utils.data.IterableDataset):
    # This converts a MapStyle dataset into an iterable one.
    # Not sure this kind of Iterable dataset is actually useful to benchmark. It
    # was necessary when benchmarking async-io stuff, but not anymore.
    # If anything, it shows how tricky Iterable datasets are to implement.
    def __init__(self, dataset, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle

        self.size = len(self.dataset)
        self.seed = 0  # has to be hard-coded for all DDP workers to have the same shuffling

    def __len__(self):
        return self.size // dist.get_world_size()

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        num_dl_workers = worker_info.num_workers
        dl_worker_id = worker_info.id

        num_ddp_workers = dist.get_world_size()
        ddp_worker_id = dist.get_rank()

        num_total_workers = num_ddp_workers * num_dl_workers
        current_worker_id = ddp_worker_id + (num_ddp_workers * dl_worker_id)

        indices = range(self.size)
        if self.shuffle:
            rng = random.Random(self.seed)
            indices = rng.sample(indices, k=self.size)
        indices = itertools.islice(indices, current_worker_id, None, num_total_workers)

        samples = (self.dataset[i] for i in indices)
        yield from samples


# TODO: maybe only generate these when --no-transforms is passed?
_RANDOM_IMAGE_TENSORS = [torch.randn(3, 224, 224) for _ in range(300)]


def no_transforms(_):
    # see --no-transforms doc
    return random.choice(_RANDOM_IMAGE_TENSORS)
