# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io

from torchdata.datapipes.iter import Filter, HuggingFaceHubReader, Mapper, Slicer, StreamReader
from torchdata.datapipes.iter.load.online import _get_response_from_http

try:
    import PIL
    from PIL import Image
except ImportError:
    PIL = None
    Image = None


# filter out images that are likely to contain watermarks
def has_no_watermark(x):
    return x["pwatermark"] is not None and x["pwatermark"] < 0.8


# filter out images that are likely to be NSFW
def is_sfw(x):
    return x["punsafe"] is not None and x["punsafe"] < 0.5


# remove images that could not be loaded
def is_working_url(x):
    return x is not None


def convert_dict_to_list(x):
    lst = list(dict.values(x))
    lst.reverse()  # StreamReader needs stream at second position
    return lst


# returns None if image could not be loaded
def load_image(url):
    try:
        timeout = 1  # 1 second timeout
        return _get_response_from_http(url, timeout=timeout)[1]  # only want the stream and not the url
    except Exception:
        return None


# For more information about the dataset see: https://laion.ai/blog/laion-5b/
# name of the dataset to be used
NAME = "laion/laion2B-en-joined"


# As the dataset is too large to store locally we use a streaming approach
def laion2b_en():
    dp = HuggingFaceHubReader(NAME)
    dp = Filter(dp, has_no_watermark)
    dp = Filter(dp, is_sfw)
    # dp = dp.shuffle().sharding_filter()
    dp = Slicer(dp, index=["TEXT", "URL"])
    dp = Mapper(dp, fn=load_image, input_col="URL")  # this desperately needs multithreading
    dp = Filter(dp, filter_fn=is_working_url, input_col="URL")
    dp = Mapper(dp, fn=convert_dict_to_list)
    dp = StreamReader(dp)
    return dp


if __name__ == "__main__":
    for i, data in enumerate(laion2b_en()):
        label, image = data
        print(f"image {i} label: {label}")
        try:
            with Image.open(io.BytesIO(image)) as img:
                try:
                    exif = img.getexif()
                    # 0x8298 is the EXIF-tag for copyright
                    copyright_info = exif.get(0x8298, "no info")
                    print(f"image {i} copyright info: {copyright_info}")
                except Exception:
                    print(f" image {i} EXIF data is corrupted")
        except PIL.UnidentifiedImageError:
            print(f"image {i} is corrupted")
