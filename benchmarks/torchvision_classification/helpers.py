import itertools
import os
import pickle
import random
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision
from PIL import Image
from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe, TarArchiveLoader


# TODO: maybe infinite buffer can / is already natively supported by torchdata?
INFINITE_BUFFER_SIZE = 1_000_000_000

IMAGENET_TRAIN_LEN = 1_281_167
IMAGENET_TEST_LEN = 50_000


class _LenSetter(IterDataPipe):
    # TODO: Ideally, we woudn't need this extra class
    def __init__(self, dp, root, args):
        self.dp = dp

        if "train" in str(root):
            if args.tiny:
                self.size = 100_000
            else:
                self.size = IMAGENET_TRAIN_LEN
        elif "val" in str(root):
            if args.tiny:
                self.size = 10_000
            else:
                self.size = IMAGENET_TEST_LEN
        else:
            raise ValueError("oops?")

    def __iter__(self):
        yield from self.dp

    def __len__(self):
        # TODO The // world_size part shouldn't be needed. See https://github.com/pytorch/data/issues/533
        if dist.is_initialized():
            return self.size // dist.get_world_size()
        else:
            return self.size


def _apply_tranforms(img_and_label, transforms):
    img, label = img_and_label
    return transforms(img), label


class ArchiveLoader(IterDataPipe):
    def __init__(self, source_datapipe, loader):
        self.loader = pickle.load if loader == "pickle" else torch.load
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for filename in self.source_datapipe:
            with open(filename, "rb") as f:
                yield self.loader(f)


class ConcaterIterable(IterDataPipe):
    # TODO: This should probably be a built-in: https://github.com/pytorch/data/issues/648
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for iterable in self.source_datapipe:
            yield from iterable


def _decode_path(data, root, category_to_int):
    path = data
    category = Path(path).relative_to(root).parts[0]
    image = Image.open(path).convert("RGB")
    label = category_to_int[category]
    return image, label


def _make_dp_from_image_folder(root):
    root = Path(root).expanduser().resolve()
    categories = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    category_to_int = {category: i for (i, category) in enumerate(categories)}

    dp = FileLister(str(root), recursive=True, masks=["*.JPEG"])

    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False).sharding_filter()
    dp = dp.map(partial(_decode_path, root=root, category_to_int=category_to_int))
    return dp


def _decode_bytesio(data):
    image, label = data
    image = Image.open(image).convert("RGB")
    return image, label


def _decode_tensor(data):
    image, label = data
    image = torchvision.io.decode_jpeg(image, mode=torchvision.io.ImageReadMode.RGB)
    return image, label


def _make_dp_from_archive(root, args):
    ext = "pt" if args.archive == "torch" else "pkl"
    dp = FileLister(str(root), masks=[f"archive_{args.archive_size}*{args.archive_content}*.{ext}"])
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False)  # inter-archive shuffling
    dp = ArchiveLoader(dp, loader=args.archive)
    dp = ConcaterIterable(dp)
    dp = dp.shuffle(buffer_size=args.archive_size).set_shuffle(False)  # intra-archive shuffling

    # TODO: we're sharding here but the big BytesIO or Tensors have already been
    # loaded by all workers, possibly in vain. Hopefully the new experimental MP
    # reading service will improve this?
    dp = dp.sharding_filter()
    decode = {"bytesio": _decode_bytesio, "tensor": _decode_tensor}[args.archive_content]
    return dp.map(decode)


def _decode_tar_entry(data):
    # Note on how we retrieve the label: each file name in the archive (the
    # "arcnames" as from the tarfile docs) looks like "label/some_name.jpg".
    # It's somewhat hacky and will obviously change, but it's OK for now.
    filename, io_stream = data
    label = int(Path(filename).parent.name)
    image = Image.open(io_stream).convert("RGB")
    return image, label


def _make_dp_from_tars(root, args):

    dp = FileLister(str(root), masks=[f"archive_{args.archive_size}*.tar"])
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False)  # inter-archive shuffling
    dp = FileOpener(dp, mode="b")
    dp = TarArchiveLoader(dp)
    dp = dp.shuffle(buffer_size=args.archive_size).set_shuffle(False)  # intra-archive shuffling
    dp = dp.sharding_filter()
    return dp.map(_decode_tar_entry)


def make_dp(root, transforms, args):
    if args.archive in ("pickle", "torch"):
        dp = _make_dp_from_archive(root, args)
    elif args.archive == "tar":
        dp = _make_dp_from_tars(root, args)
    else:
        dp = _make_dp_from_image_folder(root)

    dp = dp.map(partial(_apply_tranforms, transforms=transforms))
    dp = _LenSetter(dp, root=root, args=args)
    return dp


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


class _PreLoadedDP(IterDataPipe):
    # Same as above, but this is a DataPipe
    def __init__(self, root, transforms, buffer_size=100):
        dataset = torchvision.datasets.ImageFolder(root, transform=transforms)
        self.size = len(dataset)
        self.samples = [dataset[torch.randint(0, len(dataset), size=(1,)).item()] for i in range(buffer_size)]
        # Note: the rng might be different across DDP workers so they'll all have different samples.
        # But we don't care about accuracy here so whatever.

    def __iter__(self):
        for idx in range(self.size):
            yield self.samples[idx % len(self.samples)]


def make_pre_loaded_dp(root, transforms, args):
    dp = _PreLoadedDP(root=root, transforms=transforms)
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False).sharding_filter()
    dp = _LenSetter(dp, root=root, args=args)
    return dp


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
