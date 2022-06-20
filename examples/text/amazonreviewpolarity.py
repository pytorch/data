# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial

from torchdata.datapipes.iter import FileOpener, GDriveReader, IterableWrapper

from utils import _add_docstring_header, _create_dataset_directory, _wrap_split_argument

# URL to the target file that we will be downloading
URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"

# Expected MD5 Hash of the target file, which will be later used to verify that the file we downloaded is authentic
MD5 = "fe39f8b653cada45afd5792e0f0e8f9b"

NUM_LINES = {
    "train": 3600000,
    "test": 400000,
}

# Path/name where we will be caching the downloaded file
_PATH = "amazon_review_polarity_csv.tar.gz"

# Mapping dataset type (train/test) to the corresponding expected file names.
_EXTRACTED_FILES = {
    "train": os.path.join("amazon_review_polarity_csv", "train.csv"),
    "test": os.path.join("amazon_review_polarity_csv", "test.csv"),
}

DATASET_NAME = "AmazonReviewPolarity"


def _path_fn(root, _=None):
    return os.path.join(root, _PATH)


def _cache_path_fn(root, split, _=None):
    return os.path.join(root, _EXTRACTED_FILES[split])


def _filter_fn(split, fname_and_stream):
    return _EXTRACTED_FILES[split] in fname_and_stream[0]


def _process_tuple(t):
    return int(t[0]), " ".join(t[1:])


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def AmazonReviewPolarity(root, split):
    """Demonstrating caching, extraction and sanity check pipelines."""

    # Wrapping the URL into a IterDataPipe
    url_dp = IterableWrapper([URL])

    # `.on_disk_cache` is the functional form of `OnDiskCacheHolder`, which caches the results from the
    # subsequent DataPipe operations (until `.end_caching`) onto the disk to the path as specified by `filepath_fn`.
    # In addition, since the optional argument `hash_dict` is given, the DataPipe will also check the hashes of
    # the files before saving them. `.on_disk_cache` merely indicates that caching will take place, but the
    # content of the previous DataPipe is unchanged. Therefore, `cache_compressed_dp` still contains URL(s).
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_path_fn, root), hash_dict={_path_fn(root): MD5}, hash_type="md5"
    )

    # `GDriveReader` takes in URLs to GDrives files, and yields a tuple of file name and IO stream.
    cache_compressed_dp = GDriveReader(cache_compressed_dp)

    # `.end_caching` saves the previous DataPipe's outputs onto the disk. In this case,
    # the results from GDriveReader (i.e. the downloaded compressed archive) will be saved onto the disk.
    # Upon saving the results, the DataPipe returns the paths to the cached files.
    cache_compressed_dp = cache_compressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    # Again, `.on_disk_cache` is invoked again here and the subsequent DataPipe operations (until `.end_caching`)
    # will be saved onto the disk. At this point, `cache_decompressed_dp` contains paths to the cached files.
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(filepath_fn=partial(_cache_path_fn, root, split))

    # Opens the cache files using `FileOpener`
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b")

    # Loads the content of the TAR archive file, yielding a tuple of file names and streams of the content.
    cache_decompressed_dp = cache_decompressed_dp.load_from_tar()

    # Filters for specific file based on the file name from the previous DataPipe (either "train.csv" or "test.csv").
    cache_decompressed_dp = cache_decompressed_dp.filter(partial(_filter_fn, split))

    # ".end_caching" saves the decompressed file onto disks and yields the path to the file.
    cache_decompressed_dp = cache_decompressed_dp.end_caching(mode="wb", same_filepath_fn=True)

    # Opens the decompressed file.
    data_dp = FileOpener(cache_decompressed_dp, mode="b")

    # Finally, this parses content of the decompressed CSV file and returns the result line by line.
    return data_dp.parse_csv().map(_process_tuple)
