import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps


class RandomSingleGeometric:
    def __init__(self, probability=0.6):
        self.probability = probability

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() >= self.probability:
            return img1, img2, label1, label2

        operation = random.choice(("horizontal", "vertical", "rotate"))
        if operation == "horizontal":
            transform = Image.FLIP_LEFT_RIGHT
        elif operation == "vertical":
            transform = Image.FLIP_TOP_BOTTOM
        else:
            transform = random.choice(
                (Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)
            )

        return (
            img1.transpose(transform),
            img2.transpose(transform),
            label1.transpose(transform),
            label2.transpose(transform),
        )


class RandomScaleCrop:
    def __init__(self, probability=0.5, scale_range=(0.5, 1.5), fill=0):
        self.probability = probability
        self.scale_range = scale_range
        self.fill = fill

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() >= self.probability:
            return img1, img2, label1, label2

        width, height = img1.size
        crop_size = (width, height)
        short_size = random.randint(
            max(1, int(min(width, height) * self.scale_range[0])),
            max(1, int(min(width, height) * self.scale_range[1])),
        )

        if height > width:
            resized_width = short_size
            resized_height = int(round(height * resized_width / width))
        else:
            resized_height = short_size
            resized_width = int(round(width * resized_height / height))

        img1 = img1.resize((resized_width, resized_height), Image.BILINEAR)
        img2 = img2.resize((resized_width, resized_height), Image.BILINEAR)
        label1 = label1.resize((resized_width, resized_height), Image.NEAREST)
        label2 = label2.resize((resized_width, resized_height), Image.NEAREST)

        pad_width = max(0, crop_size[0] - resized_width)
        pad_height = max(0, crop_size[1] - resized_height)
        if pad_width or pad_height:
            border = (0, 0, pad_width, pad_height)
            img1 = ImageOps.expand(img1, border=border, fill=0)
            img2 = ImageOps.expand(img2, border=border, fill=0)
            label1 = ImageOps.expand(label1, border=border, fill=self.fill)
            label2 = ImageOps.expand(label2, border=border, fill=self.fill)

        width, height = img1.size
        max_x = max(0, width - crop_size[0])
        max_y = max(0, height - crop_size[1])
        left = random.randint(0, max_x)
        top = random.randint(0, max_y)
        right = left + crop_size[0]
        bottom = top + crop_size[1]

        box = (left, top, right, bottom)
        return img1.crop(box), img2.crop(box), label1.crop(box), label2.crop(box)


class ColorJitterSmall:
    def __init__(self, enabled=False, probability=0.5):
        self.enabled = enabled
        self.probability = probability

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if not self.enabled or random.random() >= self.probability:
            return img1, img2, label1, label2

        brightness = random.uniform(0.95, 1.05)
        contrast = random.uniform(0.95, 1.05)
        saturation = random.uniform(0.95, 1.05)
        hue = random.uniform(-0.02, 0.02)

        def apply(image):
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
            image = ImageEnhance.Color(image).enhance(saturation)
            return transforms.functional.adjust_hue(image, hue)

        return apply(img1), apply(img2), label1, label2


class RandomExchangeOrder:
    def __init__(self, probability=0.3):
        self.probability = probability

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() < self.probability:
            return img2, img1, label2, label1
        return img1, img2, label1, label2


class Normalize:
    def __init__(self, mean=None, std=None, label_threshold=128):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = None if std is None else np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.label_threshold = label_threshold

    def _normalize_image(self, image):
        image = np.asarray(image.convert("RGB"), dtype=np.float32)
        image = image.transpose((2, 0, 1)) / 255.0
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / np.maximum(self.std, 1e-6)
        return image

    def _label_to_binary(self, label):
        label = np.asarray(label)
        if label.ndim > 2:
            label = label[:, :, 2]

        binary = (label >= self.label_threshold).astype(np.float32)
        padded = np.pad(binary, 1, mode="edge")
        neighborhood_sum = sum(
            padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]]
            for dy in range(3)
            for dx in range(3)
        )
        voted = (neighborhood_sum >= 5).astype(np.float32)
        gray_mask = (label > 0) & (label < 255)
        binary[gray_mask] = voted[gray_mask]
        return binary

    def __call__(self, sample):
        if len(sample) == 4:
            img1, img2, label1, label2 = sample
            img1 = self._normalize_image(img1)
            img2 = self._normalize_image(img2)
            label1 = self._label_to_binary(label1)
            label2 = self._label_to_binary(label2)
            height, width = img1.shape[1:]
            if label1.shape != (height, width) or label2.shape != (height, width):
                raise ValueError("image and label spatial sizes do not match")
            return img1, img2, label1, label2

        if len(sample) == 2:
            img1, img2 = sample
            return self._normalize_image(img1), self._normalize_image(img2)

        raise ValueError("sample must contain two images or two images and two labels")


class ToTensor:
    def __call__(self, sample):
        return tuple(torch.from_numpy(item).float() for item in sample)


class ResizeAndPad:
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_height, self.target_width = target_size

    def __call__(self, sample):
        img1, img2 = sample
        width, height = img1.size
        scale = min(self.target_width / width, self.target_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))

        img1 = img1.resize((resized_width, resized_height), Image.BILINEAR)
        img2 = img2.resize((resized_width, resized_height), Image.BILINEAR)

        pad_width = self.target_width - resized_width
        pad_height = self.target_height - resized_height
        left = pad_width // 2
        top = pad_height // 2
        border = (left, top, pad_width - left, pad_height - top)
        return ImageOps.expand(img1, border=border, fill=0), ImageOps.expand(
            img2, border=border, fill=0
        )


def build_transforms(mean=None, std=None, color_jitter=False, input_size=256):
    normalize = Normalize(mean=mean, std=std, label_threshold=128)
    train_transforms = transforms.Compose([
        RandomSingleGeometric(probability=0.6),
        RandomScaleCrop(probability=0.5),
        ColorJitterSmall(enabled=color_jitter, probability=0.5),
        RandomExchangeOrder(probability=0.3),
        normalize,
        ToTensor(),
    ])
    eval_transforms = transforms.Compose([normalize, ToTensor()])
    inference_transforms = transforms.Compose([
        ResizeAndPad(input_size),
        normalize,
        ToTensor(),
    ])
    return train_transforms, eval_transforms, inference_transforms


with_augment_transforms, without_augment_transforms, infer_transforms = build_transforms()
