# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# From within the data directory run:
# > IMGNET_TRAIN=/path/to/imagenet/train
# > python examples/nodes/imagenet_benchmark.py --loader=process -d $IMGNET_TRAIN --max-steps 1000 --num-workers 4
#
# For FT-python, you need python 3.13t and run as:
# > python -Xgil=0 examples/nodes/imagenet_benchmark.py --loader=process -d $IMGNET_TRAIN --max-steps 1000 --num-workers 4
#
# Some example runs on Linux, with Python 3.13t below, using 4 workers
# ================================================================================
# Baseline, with torch.utils.data.DataLoader:
# > python -Xgil=1 examples/nodes/imagenet_benchmark.py --loader=classic -d $IMGNET_TRAIN --max-steps 1000 --num-workers 4
# 835.2034686705912 img/sec, 52.20021679191195 batches/sec
#
# torchdata.nodes with Multi-Processing:
# > python -Xgil=1 examples/nodes/imagenet_benchmark.py --loader=process -d $IMGNET_TRAIN --max-steps 1000 --num-workers 4
# 905.5019281357543 img/sec, 56.59387050848464 batches/sec
#
# torchdata.nodes with Multi-Threading with the GIL:
# > python -Xgil=1 examples/nodes/imagenet_benchmark.py --loader=thread -d $IMGNET_TRAIN --max-steps 1000 --num-workers 4
# 692.0924763926637 img/sec, 43.25577977454148 batches/sec
#
# torchdata.nodes with Multi-Threading with no GIL:
# > python -Xgil=0 examples/nodes/imagenet_benchmark.py --loader=thread -d $IMGNET_TRAIN --max-steps 1000 --num-workers 4
# 922.3858393659006 img/sec, 57.649114960368784 batches/sec

import argparse

import os
import time
from typing import Any, Iterator

import torch.utils.data
import torchdata.nodes as tn

import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import default_collate


class ImagenetTransform:
    """Decode, transform, and crop to 224x224.
    If called with a list of dicts, collates the results.
    """

    def __call__(self, data):
        if isinstance(data, list):
            return default_collate([self.transform_one(x) for x in data])
        else:
            return self.transform_one(data)

    def transform_one(self, data):
        img = Image.open(data["img_path"]).convert("RGB")
        img_tensor = F.pil_to_tensor(img)
        img_tensor = F.center_crop(img_tensor, [224, 224])
        data["img"] = img_tensor
        return data


class ImagenetLister:
    """Access imagenet data through either __getitem__, or an iterator.
    If using an iterator, will loop forever, in order
    """

    def __init__(self, path: str):
        self.path = path
        self.img_labels = []
        self.img_paths = []
        for label in os.listdir(path):
            for img_path in os.listdir(os.path.join(path, label)):
                self.img_labels.append(label)
                self.img_paths.append(os.path.join(path, label, img_path))

        assert len(self.img_labels) == len(self.img_paths), (
            len(self.img_labels),
            len(self.img_paths),
        )

    def __getitem__(self, i: int) -> dict:
        data = {"img_path": self.img_paths[i]}
        return data

    def __len__(self):
        return len(self.img_labels)

    def __iter__(self) -> Iterator[dict]:
        while True:  # Loop forever
            for i in range(len(self.img_labels)):
                yield {"img_path": self.img_paths[i]}


class ImagenetDataset(torch.utils.data.Dataset):
    """Classic DataLoader v1-style dataset (map style). Applies ImagenetTransform when
    retrieving items.
    """

    def __init__(self, path: str):
        self.imagenet_data = ImagenetLister(path)
        self.tx = ImagenetTransform()

    def __len__(self):
        return len(self.imagenet_data)

    def __getitem__(self, i: int) -> dict:
        return self.tx(self.imagenet_data[i])


def setup_classic(args):
    dataset = ImagenetDataset(args.imagenet_dir)
    assert args.in_order is False, "torch.utils.data.DataLoader does not support out-of-order iteration yet!"
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        shuffle=args.shuffle,
    )
    return loader


def setup(args):
    assert args.loader in ("thread", "process")
    if args.shuffle:
        dataset = ImagenetLister(args.imagenet_dir)
        sampler = torch.utils.data.RandomSampler(dataset)
        node = tn.MapStyleWrapper(map_dataset=dataset, sampler=sampler)
    else:
        node = tn.IterableWrapper(ImagenetLister(args.imagenet_dir))

    node = tn.Batcher(node, batch_size=args.batch_size)
    node = tn.ParallelMapper(
        node,
        map_fn=ImagenetTransform(),
        num_workers=args.num_workers,
        method=args.loader,
    )
    if args.pin_memory:
        node = tn.PinMemory(node)
    node = tn.Prefetcher(node, prefetch_factor=2)

    return tn.Loader(node)


def run_benchmark(args):
    print(f"Running benchmark with {args=}...")
    loader: Any
    if args.loader == "classic":
        loader = setup_classic(args)
    elif args.loader in ("thread", "process"):
        loader = setup(args)
    else:
        raise ValueError(f"Unknown loader {args.loader}")

    start = time.perf_counter()
    it = iter(loader)
    create_iter_dt = time.perf_counter() - start
    print(f"create iter took {create_iter_dt} seconds")

    start = time.perf_counter()
    if args.warmup_steps:
        for i in range(args.warmup_steps):
            next(it)
        print(f"{args.warmup_steps} warmup steps took {time.perf_counter() - start} seconds")
    warmup_dt = time.perf_counter() - start

    i: int = 0
    progress_freq = 100
    last_reported: float = time.perf_counter()
    start = time.perf_counter()
    for i in range(args.max_steps):
        if i % progress_freq == 0 or time.perf_counter() - last_reported > 5.0:
            print(f"{i} / {args.max_steps}, {time.perf_counter() - start} seconds elapsed")
            last_reported = time.perf_counter()
        next(it)
        if time.perf_counter() - start > args.max_duration:
            print(f"reached {args.max_duration=}")
            break

    iter_time = time.perf_counter() - start
    print(
        "=" * 80 + "\n"
        f"{args=}\n"
        f"Benchmark complete, {i} steps took {iter_time} seconds, "
        f"for a total of {i * args.batch_size} images\n"
        f"{i * args.batch_size / iter_time} img/sec, {i / iter_time} batches/sec\n"
        f"{create_iter_dt=}, {warmup_dt=}, {sum((create_iter_dt, warmup_dt, iter_time))=}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loader",
        default="thread",
        choices=["thread", "process", "classic"],
        help="Whether to use multi-threaded parallelism, multi-process parallelism, or the classic torch.utils.data.DataLoader (multi-process only)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers to parallelize with",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for dataloading")
    parser.add_argument("--in-order", type=bool, default=False, help="Whether to enforce ordering")
    parser.add_argument("--shuffle", type=bool, default=False, help="Whether to shuffle the data")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum number of batches to load for the benchmark",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=60,
        help="Stop after this many seconds of benchmarking, if max-steps is not reached",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of warmup steps to take before starting timing",
    )
    parser.add_argument(
        "--pin-memory",
        type=bool,
        default=False,
        help="Number of workers to parallelize with",
    )
    parser.add_argument("--imagenet-dir", "-d", type=str, required=True)
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
