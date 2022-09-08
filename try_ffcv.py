from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
from ffcv.transforms import NormalizeImage, RandomHorizontalFlip, ToTensor, ToTorchImage
import numpy as np
import time


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN = np.array(IMAGENET_MEAN) * 255
IMAGENET_STD = np.array(IMAGENET_STD) * 255
hflip_prob = 0.5
crop_size = 224


image_pipeline = [
    # SimpleRGBImageDecoder(),
    RandomResizedCropRGBImageDecoder(
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        output_size=(crop_size, crop_size),
    ),
    # RandomHorizontalFlip(hflip_prob),
    # NormalizeImage(  # Note: in original FFCV example, this is done on GPU
    #     IMAGENET_MEAN, IMAGENET_STD, np.float32
    # ),
    # ToTensor(),
    # ToTorchImage(),
]
label_pipeline = [IntDecoder()]

# Pipeline for each data field
pipelines = {
    'img': image_pipeline,
    'label': label_pipeline
}

# path = "/fsx_isolated/ktse/source_data/ffcv_data/200.beton"
# bs = 16
# num_workers = 12

# loader = Loader(path, batch_size=bs, num_workers=num_workers,
#                 order=OrderOption.QUASI_RANDOM, pipelines=pipelines,
#                 os_cache=False, drop_last=False)

# print(f"{path = }")
# print(f"{num_workers = }")

# start = time.time()
# for x in loader:
#     pass
# duration = time.time() - start
# print(f"{duration = }")

# from torchdata.datapipes.iter import IterableWrapper
# from test_imagenet import stream_to_image

# # path = "/fsx_isolated/ktse/imagenet_tars"
# # dp = IterableWrapper([path]).list_files()

# # print(list(dp))

# path = "/fsx_isolated/ktse/source_data/large_images_tars/images0.tar"
# dp = IterableWrapper([path]).open_files(mode="b").load_from_tar(mode="r:").map(stream_to_image)

# print(list(dp))


