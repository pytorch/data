import torch
from torchvision.transforms import transforms


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        on_pil_images,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        hflip_prob=0.5,
    ):
        trans = [transforms.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))

        trans.extend(
            [
                transforms.PILToTensor() if on_pil_images else torch.nn.Identity(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        on_pil_images,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor() if on_pil_images else torch.nn.Identity(),
                transforms.ConvertImageDtype(torch.float),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)
