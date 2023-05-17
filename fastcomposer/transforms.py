from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
from collections import OrderedDict


class SegmentProcessor(torch.nn.Module):
    def forward(self, image, background, segmap, id, bbox):
        mask = segmap != id
        image[:, mask] = background[:, mask]
        h1, w1, h2, w2 = bbox
        return image[:, w1:w2, h1:h2]

    def get_background(self, image):
        raise NotImplementedError


class RandomSegmentProcessor(SegmentProcessor):
    def get_background(self, image):
        background = torch.randint(
            0, 255, image.shape, dtype=image.dtype, device=image.device
        )
        return background


class PadToSquare(torch.nn.Module):
    def __init__(self, fill=0, padding_mode="constant"):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h == w:
            return image
        elif h > w:
            padding = (h - w) // 2
            image = torch.nn.functional.pad(
                image,
                (padding, padding, 0, 0),
                self.padding_mode,
                self.fill,
            )
        else:
            padding = (w - h) // 2
            image = torch.nn.functional.pad(
                image,
                (0, 0, padding, padding),
                self.padding_mode,
                self.fill,
            )
        return image


class CropTopSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h <= w:
            return image
        return image[:, :w, :]


class AlwaysCropTopSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h > w:
            return image[:, :w, :]
        else:  # h <= w
            return image[:, :, w // 2 - h // 2 : w // 2 + h // 2]


class RandomZoomIn(torch.nn.Module):
    def __init__(self, min_zoom=1.0, max_zoom=1.5):
        super().__init__()
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    def forward(self, image: torch.Tensor):
        zoom = torch.rand(1) * (self.max_zoom - self.min_zoom) + self.min_zoom
        original_shape = image.shape
        image = T.functional.resize(
            image,
            (int(zoom * image.shape[1]), int(zoom * image.shape[2])),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        # crop top square
        image = CropTopSquare()(image)
        return image


class CenterCropOrPadSides(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h > w:
            # pad sides with black
            padding = (h - w) // 2
            image = torch.nn.functional.pad(
                image,
                (padding, padding, 0, 0),
                "constant",
                0,
            )
            # resize to square
            image = T.functional.resize(
                image,
                (w, w),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            # center crop to square
            padding = (w - h) // 2
            image = image[:, :, padding : padding + h]
        return image


class TrainTransformWithSegmap(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_resize = T.Resize(
            args.train_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.segmap_resize = T.Resize(
            args.train_resolution,
            interpolation=T.InterpolationMode.NEAREST,
        )
        self.flip = T.RandomHorizontalFlip()
        self.crop = CenterCropOrPadSides()

    def forward(self, image, segmap):
        image = self.image_resize(image)
        segmap = segmap.unsqueeze(0)
        segmap = self.segmap_resize(segmap)
        image_and_segmap = torch.cat([image, segmap], dim=0)
        image_and_segmap = self.flip(image_and_segmap)
        image_and_segmap = self.crop(image_and_segmap)
        image = image_and_segmap[:3]
        segmap = image_and_segmap[3:]
        image = (image.float() / 127.5) - 1
        segmap = segmap.squeeze(0)
        return image, segmap


class TestTransformWithSegmap(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_resize = T.Resize(
            args.test_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.segmap_resize = T.Resize(
            args.test_resolution,
            interpolation=T.InterpolationMode.NEAREST,
        )
        self.crop = CenterCropOrPadSides()

    def forward(self, image, segmap):
        image = self.image_resize(image)
        segmap = segmap.unsqueeze(0)
        segmap = self.segmap_resize(segmap)
        image_and_segmap = torch.cat([image, segmap], dim=0)
        image_and_segmap = self.crop(image_and_segmap)
        image = image_and_segmap[:3]
        segmap = image_and_segmap[3:]
        image = (image.float() / 127.5) - 1
        segmap = segmap.squeeze(0)
        return image, segmap


def get_train_transforms(args):
    train_transforms = torch.nn.Sequential(
        T.Resize(
            args.train_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.RandomHorizontalFlip(),
        CenterCropOrPadSides(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5], [0.5]),
    )
    return train_transforms


def get_train_transforms_with_segmap(args):
    train_transforms = TrainTransformWithSegmap(args)
    return train_transforms


def get_test_transforms(args):
    test_transforms = torch.nn.Sequential(
        T.Resize(
            args.test_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        CenterCropOrPadSides(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5], [0.5]),
    )
    return test_transforms


def get_test_transforms_with_segmap(args):
    test_transforms = TestTransformWithSegmap(args)
    return test_transforms


def get_object_transforms(args):
    if args.no_object_augmentation:
        pre_augmentations = []
        augmentations = []
    else:
        pre_augmentations = [
            (
                "zoomin",
                T.RandomApply([RandomZoomIn(min_zoom=1.0, max_zoom=2.0)], p=0.5),
            ),
        ]

        augmentations = [
            (
                "rotate",
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=30, interpolation=T.InterpolationMode.BILINEAR
                        )
                    ],
                    p=0.75,
                ),
            ),
            ("jitter", T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.5)),
            ("blur", T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)),
            ("gray", T.RandomGrayscale(p=0.1)),
            ("flip", T.RandomHorizontalFlip()),
            ("elastic", T.RandomApply([T.ElasticTransform()], p=0.5)),
        ]

    object_transforms = torch.nn.Sequential(
        OrderedDict(
            [
                *pre_augmentations,
                ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                (
                    "resize",
                    T.Resize(
                        (args.object_resolution, args.object_resolution),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ),
                *augmentations,
                ("convert_to_float", T.ConvertImageDtype(torch.float32)),
            ]
        )
    )
    return object_transforms


def get_test_object_transforms(args):
    object_transforms = torch.nn.Sequential(
        OrderedDict(
            [
                ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                (
                    "resize",
                    T.Resize(
                        (args.object_resolution, args.object_resolution),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ),
                ("convert_to_float", T.ConvertImageDtype(torch.float32)),
            ]
        )
    )
    return object_transforms


def get_object_processor(args):
    if args.object_background_processor == "random":
        object_processor = RandomSegmentProcessor()
    else:
        raise ValueError(f"Unknown object processor: {args.object_processor}")
    return object_processor
