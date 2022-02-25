# Copyright (c) Facebook, Inc. and its affiliates.
import os.path

from torch.utils.data.datapipes.utils.decoder import imagehandler

from torchdata.datapipes.iter import FileOpener, IterableWrapper, Mapper, RoutedDecoder, TarArchiveLoader


# Download size is ~1.2 GB so fake data is provided
URL = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
ROOT = os.path.join("datasets", "caltech256")
# We really shouldn't use MD5 anymore and switch to a more secure hash like SHA256 or
# SHA512
MD5 = "67b4f42ca05d46448c6bb8ecd2220f6d"


def collate_sample(data):
    path, image = data
    dir = os.path.split(os.path.dirname(path))[1]
    label_str, cls = dir.split(".")
    return {"path": path, "image": image, "label": int(label_str), "cls": cls}


def Caltech256(root=ROOT):
    dp = IterableWrapper([os.path.join(root, "256_ObjectCategories.tar")])
    dp = FileOpener(dp, mode="b")
    dp = TarArchiveLoader(dp)
    dp = RoutedDecoder(dp, imagehandler("pil"))
    return Mapper(dp, collate_sample)


if __name__ == "__main__":
    for _sample in Caltech256():
        pass
