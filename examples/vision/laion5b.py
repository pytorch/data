# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchdata.datapipes.iter import Dropper, Filter, HuggingFaceHubReader, Mapper, Slicer
from torchdata.datapipes.iter.load.online import _get_response_from_http

try:
    import PIL
    from PIL import Image
except ImportError:
    PIL = None
    Image = None


def has_no_watermark(x):
    return x["pwatermark"] is not None and x["pwatermark"] < 0.8


def is_sfw(x):
    return x["punsafe"] is not None and x["punsafe"] < 0.5


def load_image(url):
    try:
        return _get_response_from_http(url, timeout=5)[1]
    except Exception:
        return None


def image_was_loaded(x):
    return x is not None


# For more information about the dataset see: https://laion.ai/blog/laion-5b/
# name of the dataset to be used
NAME = "laion/laion2B-en-joined"


# As the dataset is too large to store locally we use a streaming approach
def laion2b_en(name=NAME):
    dp = HuggingFaceHubReader(name)
    dp = Filter(dp, has_no_watermark)
    dp = Filter(dp, is_sfw)
    dp = dp.shuffle().sharding_filter()
    dp = Slicer(dp, index=["TEXT", "URL"])
    dp = Mapper(dp, fn=load_image, input_col="URL", output_col="IMAGE")  # this needs multithreading
    dp = Filter(dp, filter_fn=image_was_loaded, input_col="IMAGE")
    dp = Dropper(dp, "URL")
    return dp


def print_label_and_copyright(label, image):
    try:
        with Image.open(image) as img:
            try:
                exif = img.getexif()
                # 0x8298 is the EXIF-tag for copyright
                copyright_info = exif.get(0x8298, "no info")
            except Exception:
                copyright_info = "EXIF data is corrupted"
            if copyright_info != "no info" and copyright_info != "EXIF data is corrupted":
                print(f"image {i}: {label=}, {copyright_info=} ")
            else:
                print(f"image {i}: {label=}")
    except PIL.UnidentifiedImageError:
        print(f"image {i}: corrupted")


if __name__ == "__main__":
    for i, data in enumerate(laion2b_en()):
        print_label_and_copyright(data["TEXT"], data["IMAGE"])
