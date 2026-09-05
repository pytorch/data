#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark StatefulDataLoader spawn-worker startup on the public EuroSAT dataset.

This follows the benchmark shape introduced in pytorch/pytorch#159432: worker
startup, time to first batch, steady-state throughput, and process-tree memory.

Download the dataset outside the timed benchmark, then run one configuration per
fresh Python process. For example, from the TorchData repository root:

    python benchmarks/stateful_dataloader/worker_start_benchmark.py \
        --data-root /tmp/torchdata-eurosat --download --download-only
    python benchmarks/stateful_dataloader/worker_start_benchmark.py \
        --data-root /tmp/torchdata-eurosat --num-workers 8 \
        --spawn-worker-start-parallelism 1
    python benchmarks/stateful_dataloader/worker_start_benchmark.py \
        --data-root /tmp/torchdata-eurosat --num-workers 8 \
        --spawn-worker-start-parallelism 8
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import platform
import time
from typing import Any, Union

import psutil
import torch
import torchdata
import torchvision
from torch.utils.data import ConcatDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision.datasets import EuroSAT
from torchvision.io import decode_image
from torchvision.transforms import v2


def _process_tree_memory_mb() -> float:
    main_process = psutil.Process()
    processes = [main_process, *main_process.children(recursive=True)]
    memory_bytes = 0
    for process in processes:
        try:
            memory_bytes += process.memory_full_info().pss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return memory_bytes / (1024 * 1024)


def _create_dataset(
    args: argparse.Namespace,
) -> Union[EuroSAT, ConcatDataset[Any]]:
    transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = EuroSAT(
        root=args.data_root,
        transform=transform,
        download=args.download,
        loader=decode_image,
    )
    if args.dataset_copies == 1:
        return dataset
    return ConcatDataset(
        [copy.deepcopy(dataset) for _ in range(args.dataset_copies)],
    )


def _first_batch_digest(batch: tuple[torch.Tensor, torch.Tensor]) -> str:
    images, labels = batch
    digest = hashlib.sha256()
    digest.update(images.contiguous().numpy().tobytes())
    digest.update(labels.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _create_dataset(args)
    if args.download_only:
        return {"dataset_size": len(dataset), "download_only": True}

    gc.collect()
    memory_before_mb = _process_tree_memory_mb()

    constructor_start = time.perf_counter()
    loader = StatefulDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # Citrine C0: pin batches when the benchmark host has a CUDA accelerator.
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        multiprocessing_context="spawn",
        generator=torch.Generator().manual_seed(args.seed),
        spawn_worker_start_parallelism=args.spawn_worker_start_parallelism,
    )
    constructor_end = time.perf_counter()
    iterator = iter(loader)
    iterator_ready = time.perf_counter()
    first_batch = next(iterator)
    first_batch_ready = time.perf_counter()

    batches_loaded = 1
    samples_loaded = len(first_batch[1])
    peak_memory_mb = _process_tree_memory_mb()
    steady_state_start = time.perf_counter()
    while batches_loaded < args.max_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batches_loaded += 1
        samples_loaded += len(batch[1])
        if batches_loaded % 5 == 0:
            peak_memory_mb = max(peak_memory_mb, _process_tree_memory_mb())
    steady_state_seconds = time.perf_counter() - steady_state_start

    steady_state_samples = samples_loaded - len(first_batch[1])
    steady_state_samples_per_s = 0.0
    if steady_state_seconds > 0:
        steady_state_samples_per_s = steady_state_samples / steady_state_seconds
    result = {
        "batch_size": args.batch_size,
        "batches_loaded": batches_loaded,
        "constructor_s": constructor_end - constructor_start,
        "cuda_available": torch.cuda.is_available(),
        "dataset": "EuroSAT",
        "dataset_copies": args.dataset_copies,
        "dataset_size": len(dataset),
        "first_batch_digest": _first_batch_digest(first_batch),
        "iterator_creation_s": iterator_ready - constructor_end,
        "loader_and_iterator_s": iterator_ready - constructor_start,
        "memory_increase_mb": peak_memory_mb - memory_before_mb,
        "num_workers": args.num_workers,
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "platform": platform.platform(),
        "post_iterator_first_batch_s": first_batch_ready - iterator_ready,
        "python_version": platform.python_version(),
        "spawn_worker_start_parallelism": args.spawn_worker_start_parallelism,
        "steady_state_samples_per_s": steady_state_samples_per_s,
        "time_to_first_batch_s": first_batch_ready - constructor_end,
        "torch_version": torch.__version__,
        "torchdata_version": getattr(torchdata, "__version__", "source"),
        "torchvision_version": torchvision.__version__,
    }

    del iterator, loader
    gc.collect()
    return result


def _parse_args() -> argparse.Namespace:
    description = "Benchmark StatefulDataLoader spawn-worker startup on EuroSAT"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--spawn-worker-start-parallelism", type=int, default=1)
    parser.add_argument("--dataset-copies", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_workers < 1:
        raise ValueError("num_workers must be positive")
    if args.dataset_copies < 1:
        raise ValueError("dataset_copies must be positive")
    if args.max_batches < 1:
        raise ValueError("max_batches must be positive")
    print(json.dumps(_run_benchmark(args), sort_keys=True))


if __name__ == "__main__":
    main()
