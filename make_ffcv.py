import argparse
import io
import pickle
import tarfile
from math import ceil
from pathlib import Path

import torch
import torchvision
from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter
from torchvision.datasets.folder import default_loader, ImageFolder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default="/datasets01_ontap/tinyimagenet/081318/train/")
parser.add_argument("--output-path", default="./tinyimagenet/081318/train")
parser.add_argument("--archiver", default="pickle", help="pickle or tar or torch or ffcv")
parser.add_argument(
    "--archive-content",
    default="BytesIo",
    help=(
        "BytesIO or tensor or decoded. Only valid for pickle or torch archivers. "
        "However, decoded is also valid for ffcv archiver."
    ),
)
parser.add_argument(
    "--archive-size",
    type=int,
    default=500,
    help="Number of samples per archive. Not applicable for --archiver=ffcv",
)
parser.add_argument("--shuffle", type=bool, default=True, help="Whether to shuffle the samples within each archive")

# The archive parameter determines whether we use `tar.add`, `pickle.dump` or
# `torch.save` to write an archive. `torch.save` relies on pickle in the backend
# but has a special handling for tensors (which is maybe faster???):
# - tar: each tar file contains files. Each file is the original encoded jpeg
#   file. To avoid storing labels in separate files, we write the corresponding
#   label in each file name in the archive. This is ugly, but OK at this stage.
# - pickle or torch: in this case, each archive contains a list of tuples. Each
#   tuple represents a sample in the form (img_data, label). label is always an
#   int. img_data is depends on the archive-content parameter:
#   - bytesio: the *encoded* jpeg bytes which stored in a BytesIO object
#   - tensor: the *encoded* jpeg bytes which stored in a uint8 (bytes) Tensor object
#   - decoded: the *decoded* image (i.e. pixel data) in a uint8 Tensor.
#     Technically we could also store the decoded data in a BytesIO object, but
#     it's annoying as we need to keep track of the shape of the image. So it's
#     not implemented.


class Archiver:
    def __init__(
        self,
        input_dir,
        output_path,
        archiver="pickle",
        archive_content="BytesIO",
        archive_size=500,
        shuffle=True,
    ):
        self.input_dir = input_dir
        self.archiver = archiver.lower()
        self.archive_content = archive_content.lower()
        self.archive_size = archive_size
        self.shuffle = shuffle

        self.output_path = Path(output_path).resolve()
        if self.output_path.is_dir():
            self.output_path.mkdir(parents=True, exist_ok=True)

    def archive_dataset(self):
        def loader(path):
            # identity loader to avoid decoding images with PIL or something else
            # This means the dataset will always return (path_to_image_file, int_label)
            return path

        ffcv = self.archiver == "ffcv"

        dataset = ImageFolder(self.input_dir, loader=default_loader if ffcv else loader)

        if ffcv:
            return self._make_ffcv_dataset(dataset)

        self.num_samples = len(dataset)

        if self.shuffle:
            self.indices = torch.randperm(self.num_samples)
        else:
            self.indices = torch.arange(self.num_samples)

        archive_samples = []
        for i, idx in enumerate(tqdm(self.indices)):
            archive_samples.append(dataset[idx])
            if ((i + 1) % self.archive_size == 0) or (i == len(self.indices) - 1):
                archive_path = self._get_archive_path(archive_samples, last_idx=i)
                {"pickle": self._save_pickle, "torch": self._save_torch, "tar": self._save_tar}[self.archiver](
                    archive_samples, archive_path
                )

                archive_samples = []

    def _make_ffcv_dataset(self, dataset):
        DatasetWriter(
            str(self.output_path),
            {
                "img": RGBImageField(write_mode="raw" if self.archive_content == "decoded" else "jpg"),
                "label": IntField(),
            },
        ).from_indexed_dataset(dataset, shuffle_indices=self.shuffle)

    def _get_archive_path(self, samples, last_idx):
        current_archive_number = last_idx // self.archive_size
        total_num_archives_needed = ceil(self.num_samples / self.archive_size)
        zero_pad_fmt = len(str(total_num_archives_needed))
        num_samples_in_archive = len(samples)

        archive_content_str = "" if self.archiver == "tar" else f"{self.archive_content}_"
        path = (
            self.output_path
            / f"archive_{self.archive_size}_{archive_content_str}{current_archive_number:0{zero_pad_fmt}d}"
        )
        print(f"Archiving {num_samples_in_archive} samples in {path}")
        return path

    def _make_content(self, samples):
        archive_content = []
        for sample_file_name, label in samples:
            if self.archive_content == "bytesio":
                with open(sample_file_name, "rb") as f:
                    img_data = io.BytesIO(f.read())
            elif self.archive_content == "tensor":  # Note: this doesn't decode anything
                img_data = torchvision.io.read_file(sample_file_name)
            elif self.archive_content == "decoded":
                img_data = torchvision.io.read_file(sample_file_name)
                img_data = torchvision.io.decode_jpeg(img_data, mode=torchvision.io.ImageReadMode.RGB)
            else:
                raise ValueError(f"Unsupported {self.archive_content = }")
            archive_content.append((img_data, label))
        return archive_content

    def _save_pickle(self, samples, archive_name):
        archive_content = self._make_content(samples)
        archive_name = archive_name.with_suffix(".pkl")
        with open(archive_name, "wb") as f:
            pickle.dump(archive_content, f)

    def _save_torch(self, samples, archive_name):
        archive_content = self._make_content(samples)
        archive_name = archive_name.with_suffix(".pt")
        torch.save(archive_content, archive_name)

    def _save_tar(self, samples, archive_path):
        archive_path = archive_path.with_suffix(".tar")
        with tarfile.open(archive_path, "w") as tar:
            for sample_file_name, label in samples:
                path = Path(sample_file_name)
                tar.add(path, arcname=f"{label}/{path.name}")


args = parser.parse_args()
Archiver(
    input_dir=args.input_dir,
    output_path=args.output_path,
    archiver=args.archiver,
    archive_content=args.archive_content,
    shuffle=args.shuffle,
    archive_size=args.archive_size,
).archive_dataset()

# python ~/data/make_ffcv.py --input-dir ~/source_data/large_images_web --output-path ~/source_data/ffcv_data/0.beton --archiver ffcv
