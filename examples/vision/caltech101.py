# Copyright (c) Facebook, Inc. and its affiliates.
import os.path
import re

import torch
from torchdata.datapipes.iter import (
    FileOpener,
    TarArchiveReader,
    Mapper,
    RoutedDecoder,
    Filter,
    IterableWrapper,
    IterKeyZipper,
)
from torch.utils.data.datapipes.utils.decoder import imagehandler, mathandler


# Download size is ~150 MB so fake data is provided
URL = {
    "images": "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
    "annotations": "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
}
# We really shouldn't use MD5 anymore and switch to a more secure hash like SHA256 or
# SHA512
MD5 = {
    "images": "b224c7392d521a49829488ab0f1120d9",
    "annotations": "f83eeb1f24d99cab4eb377263132c91",
}

ROOT = os.path.join("fakedata", "caltech101")

IMAGES_NAME_PATTERN = re.compile(r"image_(?P<id>\d+)[.]jpg")
ANNS_NAME_PATTERN = re.compile(r"annotation_(?P<id>\d+)[.]mat")
ANNS_CLASS_MAP = {
    "Faces_2": "Faces",
    "Faces_3": "Faces_easy",
    "Motorbikes_16": "Motorbikes",
    "Airplanes_Side_2": "airplanes",
}


def is_ann(data):
    path, _ = data
    return bool(ANNS_NAME_PATTERN.match(os.path.basename(path)))


def collate_ann(data):
    path, ann = data

    cls = os.path.split(os.path.dirname(path))[1]
    if cls in ANNS_CLASS_MAP:
        cls = ANNS_CLASS_MAP[cls]

    return path, {"cls": cls, "contour": torch.as_tensor(ann["obj_contour"])}


def is_not_background_image(data):
    path, _ = data
    return os.path.split(os.path.dirname(path))[1] != "BACKGROUND_Google"


def is_not_rogue_image(data) -> bool:
    path, _ = data
    return os.path.basename(path) != "RENAME2"


def extract_file_id(path, *, pattern):
    match = pattern.match(os.path.basename(path))
    return int(match.group("id"))


def images_key_fn(data):
    path, _ = data
    cls = os.path.split(os.path.dirname(path))[1]
    id = extract_file_id(path, pattern=IMAGES_NAME_PATTERN)
    return cls, id


def anns_key_fn(data):
    path, ann = data
    id = extract_file_id(path, pattern=ANNS_NAME_PATTERN)
    return ann["cls"], id


def collate_sample(data):
    (image_path, image), (ann_path, ann) = data
    return dict(ann, image_path=image_path, image=image, ann_path=ann_path)


def Caltech101(root=ROOT):
    anns_dp = IterableWrapper([os.path.join(root, "Annotations.tar")])
    anns_dp = FileOpener(anns_dp, mode='b')
    anns_dp = TarArchiveReader(anns_dp)
    anns_dp = Filter(anns_dp, is_ann)
    anns_dp = RoutedDecoder(anns_dp, mathandler())
    anns_dp = Mapper(anns_dp, collate_ann)

    images_dp = IterableWrapper([os.path.join(root, "101_ObjectCategories.tar.gz")])
    images_dp = FileOpener(images_dp, mode='b')
    images_dp = TarArchiveReader(images_dp)
    images_dp = Filter(images_dp, is_not_background_image)
    images_dp = Filter(images_dp, is_not_rogue_image)
    images_dp = RoutedDecoder(images_dp, imagehandler("pil"))

    dp = IterKeyZipper(images_dp, anns_dp, images_key_fn, ref_key_fn=anns_key_fn, buffer_size=None)
    return Mapper(dp, collate_sample)


if __name__ == "__main__":
    for _sample in Caltech101():
        pass
