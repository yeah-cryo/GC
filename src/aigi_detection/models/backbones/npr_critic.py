"""Differentiable adapter for the official NPR deepfake detector."""
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor


def _conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = _conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, images):
        residual = images
        out = self.relu(self.bn1(self.conv1(images)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(images)
        return self.relu(out + residual)


class _NPRResNet(nn.Module):
    """The truncated ResNet-50 definition shipped by NPR."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

    def _make_layer(self, channels, blocks, stride=1):
        output_channels = channels * _Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != output_channels:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, output_channels, stride),
                nn.BatchNorm2d(output_channels),
            )
        layers = [_Bottleneck(self.inplanes, channels, stride, downsample)]
        self.inplanes = output_channels
        layers.extend(_Bottleneck(self.inplanes, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, images):
        reconstructed = F.interpolate(
            F.interpolate(images, scale_factor=.5, mode='nearest', recompute_scale_factor=True),
            scale_factor=2, mode='nearest', recompute_scale_factor=True)
        residual = images - reconstructed
        features = self.maxpool(self.relu(self.bn1(self.conv1(residual * (2 / 3)))))
        features = self.layer2(self.layer1(features))
        return self.fc1(self.avgpool(features).flatten(1)).flatten()


class NPRCritic(nn.Module):
    def __init__(self, checkpoint_path, resize_size=256, crop_size=224):
        super().__init__()
        payload = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
        state = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
        state = {name.removeprefix('module.'): value for name, value in state.items()}
        self.model = _NPRResNet()
        self.model.load_state_dict(state, strict=True)
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.eval().requires_grad_(False)

    @staticmethod
    def normalize(images):
        mean = images.new_tensor((.485, .456, .406))[None, :, None, None]
        std = images.new_tensor((.229, .224, .225))[None, :, None, None]
        return (images - mean) / std

    def preprocess(self, images):
        resized = F.interpolate(images, (self.resize_size, self.resize_size), mode='bilinear',
                                align_corners=False, antialias=True)
        offset = (self.resize_size - self.crop_size) // 2
        cropped = resized[:, :, offset:offset + self.crop_size, offset:offset + self.crop_size]
        return self.normalize(cropped)

    def forward(self, normalized_images):
        return self.model(normalized_images)

    def score_images(self, rgb_images):
        return self(self.preprocess(rgb_images))

    def score_pil(self, rgb_image: Image.Image):
        tensor = to_tensor(rgb_image).unsqueeze(0).to(next(self.parameters()).device)
        return self.score_images(tensor)
