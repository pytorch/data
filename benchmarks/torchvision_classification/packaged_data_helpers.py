import os
import pickle
from functools import partial
from pathlib import Path

import torch
from helpers import _apply_tranforms, _decode, _LenSetter as LenSetter
from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe, TarArchiveLoader


INFINITE_BUFFER_SIZE = 1_000_000_000


def _drop_label(data):
    img_data, label = data
    return img_data


def _read_tar_entry(data):
    _, io_stream = data
    return io_stream.read()


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

    dp = FileLister(str(root), masks=[f"archive_{archive_size}*.tar"])
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE)  # inter-archive shuffling
    dp = FileOpener(dp, mode="b")
    dp = TarArchiveLoader(dp, mode="r:")
    dp = dp.shuffle(buffer_size=archive_size)  # intra-archive shuffling
    dp = dp.sharding_filter()

    dp = dp.map(_read_tar_entry)
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
    dp = dp.shuffle(buffer_size=INFINITE_BUFFER_SIZE)  # inter-archive shuffling
    dp = ArchiveLoader(dp, loader=archive)
    dp = ConcaterIterable(dp)
    dp = dp.map(_drop_label)
    dp = dp.shuffle(buffer_size=archive_size)  # intra-archive shuffling

    # TODO: we're sharding here but the big BytesIO or Tensors have already been
    # loaded by all workers, possibly in vain. Hopefully the new experimental MP
    # reading service will improve this?
    dp = dp.sharding_filter()
    return dp


def make_dp_from_packaged_data(*, root, archive=None, archive_content=None, archive_size=500, transforms=None):
    if archive in ("pickle", "torch"):
        dp = _make_dp_from_archive(
            root=root, archive=archive, archive_content=archive_content, archive_size=archive_size
        )
    elif archive == "tar":
        dp = _make_dp_from_tars(root=root, archive_size=archive_size)
    # elif archive == "webdataset":
    #     dp = _make_webdataset(root=root, archive_size=archive_size)
    else:  # archive == `None`
        dp = _make_dp_from_image_folder(root=root)

    # Decode
    root = Path(root).expanduser().resolve()
    categories = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    category_to_int = {category: i for (i, category) in enumerate(categories)}
    dp = dp.map(partial(_decode, root=root, category_to_int=category_to_int))

    if dp is not None:
        dp = dp.map(partial(_apply_tranforms, transforms=transforms))

    dp = LenSetter(dp=dp, root=root)
    return dp
