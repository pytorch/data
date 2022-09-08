import io

import numpy as np
import torch
from common import bench, decode, JPEG_FILES_ROOT
from ffcv import libffcv
from PIL import Image
from torchvision.io import read_file

torch.set_num_threads(1)
files = list((JPEG_FILES_ROOT / ("n01629819/" if args.tiny else "n02492035/")).glob("*.JPEG"))

tensors = [read_file(str(filepath)) for filepath in files]
np_arrays = [t.numpy().astype(np.uint8) for t in tensors]

bytesio_list = []
for filepath in files:
    with open(filepath, "rb") as f:
        bytesio_list.append(io.BytesIO(f.read()))

decoded_tensors = [decode(t) for t in tensors]
sizes = [tuple(t.shape[1:]) for t in decoded_tensors]
dests = [np.zeros(h * w * 3, dtype=np.uint8) for (h, w) in sizes]


def decode_turbo(arr, h, w, dest):
    libffcv.imdecode(arr, dest, h, w, h, w, 0, 0, 1, 1, False, False)


# if __name__ == "__main__":

#     unit = "μ" if args.tiny else "m"
#     print("PIL.Image.open(bytesio).load()")
#     bench(
#         lambda l: [Image.open(bytesio).convert("RGB").load() for bytesio in l],
#         bytesio_list,
#         unit=unit,
#         num_images_per_call=len(bytesio_list),
#     )

#     print("decode_jpeg(tensor)")
#     bench(lambda l: [decode(t) for t in l], tensors, unit=unit, num_images_per_call=len(tensors))

#     print("libffcv.imdecode - using libjpeg-turbo")
#     bench(
#         (lambda l: [decode_turbo(arr, h, w, dest) for (arr, (h, w), dest) in l]),
#         list(zip(np_arrays, sizes, dests)),
#         unit=unit,
#         num_images_per_call=len(np_arrays),
#     )
#     print()
