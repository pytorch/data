import argparse
import os

import torch
from torch import distributed as dist

from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.ibm_rescalable import (
    DummyDataset,
    PreprocessDataset,
    SamplingDataset,
    ScalableShardDataset,
    load_distributed_state_dict,
    save_distributed_state_dict,
)

# This example script validates the rescaling behavior of the ibm rescalable distributed datasets.
# On first run, saves a distributed checkpoint to the desired location.
# On subsequent runs, loads the checkpoint (possibly on a different world size / num workers)
# and verifies that previous data is not revisited, while upcoming data is.

# Example usage:
# torchrun [torchrun args] examples/ibm_rescaling/rescaling_demo.py --ckpt_path=~/ckpts/rescale_test --logical_shards=48 --num_workers=6


parser = argparse.ArgumentParser(description="Script to validate rescaling of dataloader checkpoints")
parser.add_argument("--ckpt_path", type=str, default="./rescale_test")
parser.add_argument(
    "--logical_shards",
    type=int,
    default=96,
    help="Total number of data partitions. (worldsize * n_workers) must divide this evenly.",
)
parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers per device")
parser.add_argument("--b_size", type=int, default=1, help="Number of data points per step per device")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# Setup
rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
dist.init_process_group()
mesh = dist.device_mesh.init_device_mesh("cpu", [world_size])
placement = [dist.tensor.placement_types.Shard(0)]

# Build dataloader
data = DummyDataset(None, rank, world_size, delimiter_token=-1, seed=args.seed)
# Pretend that we're sampling over multiple sub-datasets
subdatas = ["sub_dataset", "second_subdataset", "small_subdataset"]
[os.makedirs(os.path.join(args.ckpt_path, "data", subdata), exist_ok=True) for subdata in subdatas]
data = SamplingDataset(
    os.path.join(args.ckpt_path, "data"),
    data,
    delimiter_token=-1,
    datasets=subdatas,
    weights=[12, 17, 5],
)
# Apply rescalability layer
data = ScalableShardDataset(data, n_logical_shards=args.logical_shards)
# Statelessly convert all outputs to tensors
data = PreprocessDataset(data, torch.tensor)
# Wrap in StatefulDataLoader
data = StatefulDataLoader(data, batch_size=args.b_size, num_workers=args.num_workers)

# If checkpoint does not exist, create it
if not os.path.exists(args.ckpt_path) or len(os.listdir(cfg.ckpt_save_path)) == 0:
    os.makedirs(args.ckpt_path, exist_ok=True)
    # Iterate, assemble values to exclude
    if rank == 0:
        print("No existing checkpoint. Processing 100 steps.")

    avoid = []
    for i, inp in enumerate(data):
        if i == 100:
            if rank == 0:
                print("Iteration complete!")
            save_distributed_state_dict(data, os.path.join(args.ckpt_path, "loader_dcp_state"), mesh)
            break
        avoid.append(inp)
    avoid = torch.cat(avoid)
    # Get all vals onto each rank
    avoid = dist.tensor.DTensor.from_local(
        avoid,
        mesh,
        placement,
    ).full_tensor()

    # Continue, assemble values to include
    load_distributed_state_dict(data, os.path.join(args.ckpt_path, "loader_dcp_state"), mesh)
    if rank == 0:
        print("DCP state loaded!")

    include = []
    for i, inp in enumerate(data):
        if i == 10:
            break
        include.append(inp)
    include = torch.cat(include)
    if rank == 0:
        print("Iteration round 2 complete!")
    # Get all vals onto each rank
    include = dist.tensor.DTensor.from_local(include, mesh, placement).full_tensor()

    if rank == 0:
        torch.save(avoid, os.path.join(args.ckpt_path, "avoid.pth"))
        torch.save(include, os.path.join(args.ckpt_path, "include.pth"))
        print(
            "Generation complete! Please rerun (with different world size / workers if desired) to complete the check."
        )

# If checkpoint does exist, load and take 100 steps.
# Ensure avoid values are avoided, and all include values are included.
else:
    if rank == 0:
        print("Checkpoint detected!")
    load_distributed_state_dict(data, os.path.join(args.ckpt_path, "loader_dcp_state"), mesh)

    vals = []
    for i, inp in enumerate(data):
        if i == 100:
            break
        vals.append(inp)
    vals = torch.cat(vals)
    # Get all vals onto each rank
    vals = dist.tensor.DTensor.from_local(vals, mesh, placement).full_tensor()

    # Perform avoid/include checks on rank 0 only
    if rank == 0:
        avoid = torch.load(os.path.join(args.ckpt_path, "avoid.pth"))
        include = torch.load(os.path.join(args.ckpt_path, "include.pth"))

        def _in(v, m):
            # Returns whether vector v is a row of matrix m (both tensors)
            return m.sub(v[None]).abs().sum(1).sign().prod().bool().logical_not().item()

        # Avoid check
        for i, x in enumerate(avoid.split(1)):
            assert not _in(x[0], vals), i
        print("Check passed: seen data was not revisited!")

        # Include check
        for i, x in enumerate(include.split(1)):
            assert _in(x[0], vals), i
        print("Check passed: upcoming data appears as expected!")

dist.barrier()
dist.destroy_process_group()
