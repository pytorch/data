# import os
import pickle
from functools import partial
from pathlib import Path

import torch
from helpers import _apply_tranforms, _LenSetter as LenSetter
from PIL import Image
from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe, TarArchiveLoader
from torchvision import transforms
from torchvision.io import decode_jpeg, ImageReadMode


INFINITE_BUFFER_SIZE = 1_000_000_000


def _drop_label(data):
    img_data, label = data
    return img_data


def _read_tar_entry(data):
    _, io_stream = data
    return io_stream.read()


def bytesio_to_tensor(bytesio):
    return torch.frombuffer(bytesio.getbuffer(), dtype=torch.uint8)


def decode(encoded_tensor):
    try:
        return decode_jpeg(encoded_tensor, mode=ImageReadMode.RGB)
    except RuntimeError:
        # Happens in ImageNet for ~20 CYMK images that can't be decoded with decode_jpeg()
        # Asking for forgivness is better than blahblahlblah... BTW, hard
        # disagree on this but anyway, the addition of the try/except statement
        # doesn't impact benchmark results significantly, so we're fine.
        return transforms.PILToTensor()(Image.fromarray(encoded_tensor.numpy()).convert("RGB"))


class ConcaterIterable(IterDataPipe):
    # TODO: This should probably be a built-in: https://github.com/pytorch/data/issues/648
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for iterable in self.source_datapipe:
            yield from iterable


class ArchiveLoader(IterDataPipe):
    def __init__(self, source_datapipe, loader):
        self.loader = pickle.load if loader == "pickle" else torch.load
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for filename in self.source_datapipe:
            with open(filename, "rb") as f:
                yield self.loader(f)


def _make_dp_from_tars(*, root, archive_size):

    print("in _make_dp_from_tars")
    dp = FileLister(str(root), masks=[f"archive_{archive_size}*.tar"])
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE)  # inter-archive shuffling
    dp = FileOpener(dp, mode="b")
    dp = TarArchiveLoader(dp, mode="r:")
    dp = dp.shuffle(buffer_size=archive_size)  # intra-archive shuffling
    dp = dp.sharding_filter()

    dp = dp.map(_read_tar_entry)

    # TODO: Figure out how to decode and transform these
    dp = dp.header(20)
    print(list(dp))
    return dp


def _make_dp_from_image_folder(*, root):
    dp = FileLister(str(root), recursive=True, masks=["*.JPEG"])
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE).sharding_filter()
    return dp


def _make_dp_from_archive(*, root, archive, archive_content, archive_size):
    """
    This works for `pickle` and `torch`
    """
    ext = "pt" if archive == "torch" else "pkl"
    dp = FileLister(str(root), masks=[f"archive_{archive_size}*{archive_content}*.{ext}"])
    print(f"FileLister {dp}")
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE)  # inter-archive shuffling
    dp = ArchiveLoader(dp, loader=archive)
    dp = ConcaterIterable(dp)
    dp = dp.map(_drop_label)
    print(f"dp.map(_drop_label) {dp}")

    dp = dp.shuffle(buffer_size=archive_size)  # intra-archive shuffling

    # TODO: we're sharding here but the big BytesIO or Tensors have already been
    # loaded by all workers, possibly in vain. Hopefully the new experimental MP
    # reading service will improve this?
    dp = dp.sharding_filter()
    return dp


def make_dp_from_packaged_data(*, root, archive=None, archive_content=None, archive_size=500, transforms=None):
    if archive in ("pickle", "torch"):  # Assume to be BytesIO
        dp = _make_dp_from_archive(
            root=root, archive=archive, archive_content=archive_content, archive_size=archive_size
        )
        dp = dp.map(bytesio_to_tensor).map(decode)
    elif archive == "tar":
        dp = _make_dp_from_tars(root=root, archive_size=archive_size)
    # elif archive == "webdataset":
    #     dp = _make_webdataset(root=root, archive_size=archive_size)
    else:  # archive == `None`
        dp = _make_dp_from_image_folder(root=root)

    # Decode
    root = Path(root).expanduser().resolve()
    # categories = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    # category_to_int = {category: i for (i, category) in enumerate(categories)}
    # dp = dp.map(partial(_decode, root=root, category_to_int=category_to_int))

    if transforms is not None:
        dp = dp.map(partial(_apply_tranforms, transforms=transforms))

    dp = LenSetter(dp=dp, root=root)
    return dp
