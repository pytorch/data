import argparse
import contextlib
import datetime
import traceback
from pathlib import Path
from time import perf_counter

import torch
import webdataset as wds

from ffcv.loader import Loader as FFCVLoader
from PIL import Image
from torch.utils import data
from torchdata.dataloader2 import DataLoader2
from torchvision import transforms

from torchvision.datasets import ImageFolder
from torchvision.io import decode_jpeg, ImageReadMode


# parser = argparse.ArgumentParser()
# parser.add_argument("--fs", default="fsx_isolated")
# parser.add_argument("--tiny", action="store_true")
# parser.add_argument("--num-workers", type=int, default=0)
# parser.add_argument("--limit", type=int, default=None, help="Load at most `limit` samples")
# parser.add_argument("--archive-size", type=int, default=500, help="Number of samples in each archive.")
# args = parser.parse_args()

# print(args)

# path_to_dataset = "tinyimagenet/081318/" if args.tiny else "imagenet_full_size/061417"
# COMMON_ROOT = Path("/") / args.fs / "nicolashug" / path_to_dataset
# ARCHIVE_ROOT = COMMON_ROOT / "archives/train"
# JPEG_FILES_ROOT = COMMON_ROOT / "train"

# DATASET_SIZE = 100_000 if args.tiny else 1_281_167

# Deactivate OMP / MKL parallelism: in most cases we'll run the data-loading
# pipeline within a parallelized DataLoader which will call this as well anyway.
torch.set_num_threads(1)


def bytes_to_image(bytes):
    # return Image.open(StringIO(bytes)).convert("RGB")
    return Image.open(bytes).convert("RGB")


def bench(f, inp, num_exp=3, warmup=1, unit="μ", num_images_per_call=None):
    if num_images_per_call is None:
        num_images_per_call = args.limit or DATASET_SIZE

    # Computes PER IMAGE median times
    for _ in range(warmup):
        f(inp)

    times = []
    for _ in range(num_exp):
        start = perf_counter()
        f(inp)
        end = perf_counter()
        times.append((end - start))

    mul = {"μ": 1e6, "m": 1e3, "s": 1}[unit]
    times = torch.tensor(times) / num_images_per_call
    median_sec = torch.median(times)

    times_unit = times * mul
    median_unit = torch.median(times_unit)

    over_10_epochs = datetime.timedelta(seconds=int(median_sec * DATASET_SIZE * 10))

    s = f"{median_unit:.1f} {unit}{'s' if unit != 's' else ''}/img (std={torch.std(times_unit):.2f})"
    print(f"{s:30}   {int(1 / median_sec):15,}   {over_10_epochs}")
    print()
    return median_sec


def iterate_one_epoch(obj):
    if isinstance(
        obj,
        (
            data.datapipes.datapipe.IterDataPipe,
            data.DataLoader,
            DataLoader2,
            wds.WebLoader,
            wds.WebDataset,
        ),
    ):
        for _ in obj:
            pass
    elif isinstance(obj, ImageFolder):
        # Need to reproduce "random" access
        indices = torch.randperm(len(obj))
        for i in indices:
            obj[i]
    elif isinstance(obj, FFCVLoader):
        if args.limit is not None:
            limit = args.limit // obj.batch_size
        else:
            limit = len(obj)
        i = 0
        for i, _ in enumerate(obj):
            if i == limit:
                break
    else:
        raise ValueError("Ugh?")


def decode(encoded_tensor):
    try:
        return decode_jpeg(encoded_tensor, mode=ImageReadMode.RGB)
    except RuntimeError:
        # Happens in ImageNet for ~20 CYMK images that can't be decoded with decode_jpeg()
        # Asking for forgivness is better than blahblahlblah... BTW, hard
        # disagree on this but anyway, the addition of the try/except statement
        # doesn't impact benchmark results significantly, so we're fine.
        return transforms.PILToTensor()(Image.fromarray(encoded_tensor.numpy()).convert("RGB"))


def bytesio_to_tensor(bytesio):
    return torch.frombuffer(bytesio.getbuffer(), dtype=torch.uint8)


@contextlib.contextmanager
def suppress():
    # Like contextlib.suppress(Exception), but prints the exception as well
    try:
        yield
    except Exception as exc:
        print("This raised the following exception:")
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        print("Continuing as if nothing happened...")
