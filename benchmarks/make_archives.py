import argparse
import io
import pickle
import tarfile
from math import ceil
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default="/datasets01_ontap/tinyimagenet/081318/train/")
parser.add_argument("--output-dir", default="./tinyimagenet/081318/train")
parser.add_argument("--archiver", default="pickle", help="pickle or tar or torch")
parser.add_argument(
    "--archive-content", default="BytesIo", help="BytesIO or tensor. Only valid for pickle or torch archivers"
)
parser.add_argument("--archive-size", type=int, default=500, help="Number of samples per archive")
parser.add_argument("--shuffle", type=bool, default=True, help="Whether to shuffle the samples within each archive")

# The archive parameter determines whether we use `tar.add`, `pickle.dump` or
# `torch.save` to write an archive. `torch.save` relies on pickle in the backend
# but has a special handling for tensors (which is maybe faster???):
# - tar: each tar file contains files. Each file is the original encoded jpeg
#   file. To avoid storing labels in separate files, we write the corresponding
#   label in each file name in the archive. This is ugly, but OK at this stage.
# - pickle or torch: in this case, each archive contains a list of tuples. Each
#   tuple represents a sample in the form (img_data, label). label is always an
#   int, and img_data is the *encoded* jpeg bytes which can be represented
#   either as a tensor or a BytesIO object, depending on the archive-content
#   parameter.


class Archiver:
    def __init__(
        self,
        input_dir,
        output_dir,
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

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def archive_dataset(self):
        def loader(path):
            # identity loader to avoid decoding images with PIL or something else
            # This means the dataset will always return (path_to_image_file, int_label)
            return path

        dataset = torchvision.datasets.ImageFolder(self.input_dir, loader=loader)
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

    def _get_archive_path(self, samples, last_idx):
        current_archive_number = last_idx // self.archive_size
        total_num_archives_needed = ceil(self.num_samples / self.archive_size)
        zero_pad_fmt = len(str(total_num_archives_needed))
        num_samples_in_archive = len(samples)

        archive_content_str = "" if self.archiver == "tar" else f"{self.archive_content}_"
        path = (
            self.output_dir
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
    output_dir=args.output_dir,
    archiver=args.archiver,
    archive_content=args.archive_content,
    shuffle=args.shuffle,
    archive_size=args.archive_size,
).archive_dataset()
