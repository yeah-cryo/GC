"""Differentiable adapter for the released SAFE wavelet detector."""
from pathlib import Path

import torch
from PIL import Image
from pytorch_wavelets import DWTForward
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor

from .npr_critic import _NPRResNet


class _SAFEBackbone(_NPRResNet):
    """SAFE and NPR share the same truncated ResNet body but use different inputs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2)

    def forward(self, high_frequency_images):
        features = self.maxpool(
            self.relu(self.bn1(self.conv1(high_frequency_images))))
        features = self.layer2(self.layer1(features))
        return self.fc1(self.avgpool(features).flatten(1))


class SAFECritic(nn.Module):
    def __init__(self, checkpoint_state, crop_size=256):
        super().__init__()
        state = checkpoint_state['model'] if 'model' in checkpoint_state else checkpoint_state
        self.model = _SAFEBackbone()
        self.model.load_state_dict(state, strict=True)
        self.wavelet = DWTForward(J=1, mode='symmetric', wave='bior1.3')
        self.crop_size = int(crop_size)
        self.eval().requires_grad_(False)

    def preprocess(self, images):
        height, width = images.shape[-2:]
        if min(height, width) < self.crop_size:
            dy, dx = max(0, self.crop_size - height), max(0, self.crop_size - width)
            images = F.pad(images, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
            height, width = images.shape[-2:]
        top = (height - self.crop_size) // 2
        left = (width - self.crop_size) // 2
        cropped = images[:, :, top:top + self.crop_size, left:left + self.crop_size]
        _, high = self.wavelet(cropped.float())
        diagonal = high[0][:, :, 2]
        return F.interpolate(diagonal, (self.crop_size, self.crop_size), mode='bilinear',
                             align_corners=False, antialias=True)

    def forward(self, high_frequency_images):
        logits = self.model(high_frequency_images)
        return (logits[:, 1] - logits[:, 0]).float()

    def score_images(self, rgb_images):
        return self(self.preprocess(rgb_images))

    def score_pil(self, rgb_image: Image.Image):
        tensor = to_tensor(rgb_image).unsqueeze(0).to(next(self.parameters()).device)
        return self.score_images(tensor)
