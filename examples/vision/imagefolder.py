# Copyright (c) Facebook, Inc. and its affiliates.
import os
import re
import http.server
import threading

import torch
import torch.utils.data.backward_compatibility
import torchvision.datasets as datasets
import torchvision.datasets.folder
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import HttpReader, FileLister, IterDataPipe

IMAGES_ROOT = os.path.join("fakedata", "imagefolder")

USE_FORK_DATAPIPE = False
NUM_WORKERS = 5
BATCH_SIZE = None

data_transform = transforms.Compose(
    [
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# DataPipes implementation of ImageFolder constructs and executes graph of DataPipes (aka DataPipeline)
# FileLister -> ObtainCategories
#                       |
#                       V
# FileLister -> AttributeCategories -> LoadAndDecodeImages (using `map`) -> ApplyTorchVisionTransforms (using `map`)


def get_category_name(path):
    rel_path = os.path.relpath(path, start=IMAGES_ROOT)
    elements = rel_path.split(os.sep)
    return elements[0]


class ObtainCategories(IterDataPipe):
    def __init__(self, source_dp, parse_category_fn=get_category_name):
        self.source_dp = source_dp
        self.parse_category_fn = parse_category_fn

    def __iter__(self):
        categories = set()
        for path in self.source_dp:
            categories.add(self.parse_category_fn(path))
        cat_to_id = {name: i for i, name in enumerate(sorted(categories))}
        yield cat_to_id


class AttributeCategories(IterDataPipe):
    def __init__(self, listfiles_dp, categories_dp, parse_category_fn=get_category_name):
        self.listfiles_dp = listfiles_dp
        self.categories_dp = categories_dp
        self.parse_category_fn = parse_category_fn

    def __iter__(self):
        for categories in self.categories_dp:
            cat_to_dp = categories
        for data in self.listfiles_dp:
            if isinstance(data, tuple):
                category = cat_to_dp[self.parse_category_fn(data[0])]
                ct = tuple([category])
                yield data + ct
            else:
                category = cat_to_dp[self.parse_category_fn(data)]
                ct = tuple([category])
                data = tuple([data])
                yield data + ct


def MyImageFolder(root=IMAGES_ROOT, transform=None):
    if not USE_FORK_DATAPIPE:
        # Yes, we had to scan files twice. Alternativelly it is possible to use
        # `fork` DataPipe, but it will require buffer equal to the size of all
        # full file names
        # TODO(VitalyFedyunin): Make sure that `fork` complains when buffer becomes
        # too large
        list_files_0 = FileLister(root=IMAGES_ROOT, recursive=True)
        list_files_1 = FileLister(root=IMAGES_ROOT, recursive=True).sharding_filter()
    else:
        list_files_0, list_files_1 = FileLister(root=IMAGES_ROOT, recursive=True).fork(2)
        list_files_1 = list_files_1.sharding_filter()

    categories = ObtainCategories(list_files_0)
    with_categories = AttributeCategories(list_files_1, categories)
    using_default_loader = with_categories.map(lambda x: (torchvision.datasets.folder.default_loader(x[0]), x[1]))
    transformed = using_default_loader.map(lambda x: (transform(x[0]), x[1]))
    return transformed


class ExpandURLPatternDataPipe(IterDataPipe):
    def __init__(self, pattern):
        result = re.match(r"(.*?)\{(.*?)}(.*)", pattern)
        if result:
            self.prefix = result.group(1)
            self.pattern = result.group(2)
            self.postfix = result.group(3)
            result = re.match(r"(\d+)\.\.(\d+)", self.pattern)
            if result:
                self.start_str = result.group(1)
                self.end_str = result.group(2)
            else:
                raise Exception("Invalid pattern")
        else:
            raise Exception("Invalid pattern")

    def __iter__(self):
        current_int = int(self.start_str)
        end_int = int(self.end_str)
        for i in range(current_int, end_int + 1):
            str_i = str(i)
            while len(str_i) < len(self.start_str):
                str_i = "0" + str_i
            yield self.prefix + str_i + self.postfix


HTTP_PATH_ROOT = "http://localhost:8000/"
HTTP_PATH_CAT = "http://localhost:8000/cat/{1..3}.jpg"
HTTP_PATH_DOG = "http://localhost:8000/dog/{1..3}.jpg"


def get_category_name_url(url):
    rel_path = os.path.relpath(url, start=HTTP_PATH_ROOT)
    elements = rel_path.split(os.sep)
    return elements[0]


def stream_to_pil(stream):
    img = Image.open(stream)
    return img.convert("RGB")


def MyHTTPImageFolder(transform=None):
    # HTTP Protocol doesn't support listing files, so we had to provide it explicitly
    list_files = ExpandURLPatternDataPipe(HTTP_PATH_CAT) + ExpandURLPatternDataPipe(HTTP_PATH_DOG)

    list_files_0, list_files_1 = list_files.fork(2)
    list_files_1 = list_files_1.sharding_filter().shuffle()

    categories = ObtainCategories(list_files_0, parse_category_fn=get_category_name_url)

    loaded_files = HttpReader(list_files_1)

    with_categories = AttributeCategories(loaded_files, categories, parse_category_fn=get_category_name_url)
    pil_images = with_categories.map(lambda x: (x[0], stream_to_pil(x[1]), x[2]))
    transformed = pil_images.map(lambda x: (transform(x[1]), x[2]))
    return transformed


if __name__ == "__main__":
    dataset = datasets.ImageFolder(root=IMAGES_ROOT, transform=data_transform)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    items = list(dl)
    assert len(items) == 6

    dataset = MyImageFolder(root=IMAGES_ROOT, transform=data_transform)
    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=torch.utils.data.backward_compatibility.worker_init_fn,
    )
    items = list(dl)
    assert len(items) == 6

    http_handler = http.server.SimpleHTTPRequestHandler
    http_handler.log_message = lambda a, b, c, d, e: None
    httpd = http.server.HTTPServer(("", 8000), http_handler)
    os.chdir(IMAGES_ROOT)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()

    dataset = MyHTTPImageFolder(transform=data_transform)
    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=torch.utils.data.backward_compatibility.worker_init_fn,
    )

    try:
        items = list(dl)
        assert len(items) == 6
    finally:
        httpd.shutdown()
