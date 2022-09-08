import torch

import torchvision.transforms as transforms
# from bench_decoding import bytesio_list, decoded_tensors
# from common import bench
# from PIL import Image


class ToContiguous(torch.nn.Module):
    # Can't be lambda otherwise datapipes fail
    def forward(self, x):
        return x.contiguous()


class ClassificationPresetTrain:
    def __init__(self, *, on):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        hflip_prob = 0.5
        crop_size = 224
        on = on.lower()
        if on not in ("tensor", "pil"):
            raise ValueError("oops")

        trans = []

        if on == "tensor":
            trans += [ToContiguous()]

        trans += [transforms.RandomResizedCrop(crop_size, antialias=True)]
        if hflip_prob > 0:
            trans += [transforms.RandomHorizontalFlip(hflip_prob)]

        if on == "pil":
            trans += [transforms.PILToTensor()]

        trans += [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


# if __name__ == "__main__":

#     print("PIL tranforms")
#     pil_imgs = [Image.open(bytesio).convert("RGB") for bytesio in bytesio_list]
#     bench(
#         lambda l: [ClassificationPresetTrain(on="PIL")(img) for img in l],
#         pil_imgs,
#         unit="m",
#         num_images_per_call=len(pil_imgs),
#     )

#     print("Tensor transforms")
#     bench(
#         lambda l: [ClassificationPresetTrain(on="tensor")(t) for t in l],
#         decoded_tensors,
#         unit="m",
#         num_images_per_call=len(decoded_tensors),
#     )
#     print()
