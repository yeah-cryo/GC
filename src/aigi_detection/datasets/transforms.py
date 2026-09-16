import io
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def jpeg(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert('RGB')


def pad(image, size=224):
    width, height = image.size
    dx, dy = max(0, size - width), max(0, size - height)
    return ImageOps.expand(image, (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2), fill=0)


def normalize(image):
    return F.normalize(F.to_tensor(image), MEAN, STD)


class TrainTransform:
    def __init__(self, size=224):
        self.size = size
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

    def __call__(self, image, label):
        if random.random() < 0.5:
            scale = random.uniform(0.5, 2.0)
            image = image.resize(tuple(max(1, round(v * scale)) for v in image.size), Image.Resampling.BILINEAR)
        if random.random() < 0.5:
            image = jpeg(image, random.randint(50, 100))
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0, 3)))
        if random.random() < 0.5:
            pixels = np.asarray(image).astype(np.float32)
            pixels += np.random.normal(0, random.uniform(0, 55), pixels.shape).astype(np.float32)
            image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        image = self.jitter(image)
        if random.random() < 0.5:
            image = F.hflip(image)
        image = F.rotate(image, random.uniform(-10, 10), interpolation=T.InterpolationMode.BILINEAR, fill=0)
        image = pad(image, self.size)
        image = F.crop(image, *T.RandomCrop.get_params(image, (self.size, self.size)))
        return normalize(image)


class ValidationTransform:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image, label):
        if label == 1:
            image = jpeg(image, 96)
        crops = F.five_crop(pad(image, self.size), self.size)
        return torch.stack([normalize(crop) for crop in crops])
