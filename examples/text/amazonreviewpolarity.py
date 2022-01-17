# Copyright (c) Facebook, Inc. and its affiliates.
import os

from torchdata.datapipes.iter import FileOpener, GDriveReader, IterableWrapper

from .utils import _add_docstring_header, _create_dataset_directory, _wrap_split_argument


URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"

MD5 = "fe39f8b653cada45afd5792e0f0e8f9b"

NUM_LINES = {
    "train": 3600000,
    "test": 400000,
}

_PATH = "amazon_review_polarity_csv.tar.gz"

_EXTRACTED_FILES = {
    "train": os.path.join(_PATH, "amazon_review_polarity_csv", "train.csv"),
    "test": os.path.join(_PATH, "amazon_review_polarity_csv", "test.csv"),
}

_EXTRACTED_FILES_MD5 = {
    "train": "520937107c39a2d1d1f66cd410e9ed9e",
    "test": "f4c8bded2ecbde5f996b675db6228f16",
}

DATASET_NAME = "AmazonReviewPolarity"


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def AmazonReviewPolarity(root, split):
    """Demonstrating caching, extraction and sanity check pipelines."""

    url_dp = IterableWrapper([URL])
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, _PATH), hash_dict={os.path.join(root, _PATH): MD5}, hash_type="md5"
    )
    cache_compressed_dp = GDriveReader(cache_compressed_dp).end_caching(mode="wb", same_filepath_fn=True)
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda x: os.path.join(root, os.path.dirname(_EXTRACTED_FILES[split]), os.path.basename(x)))
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").read_from_tar()
    cache_compressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)
    data_dp = FileOpener(cache_decompressed_dp.filter(lambda x: _EXTRACTED_FILES[split] in x[0]).map(lambda x: x[0]), mode='b')
    return data_dp.parse_csv().map(fn=lambda t: (int(t[0]), ' '.join(t[1:])))
