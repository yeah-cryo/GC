"""Differentiable adapter for a trained SimLBR DINOv3-L classifier."""
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_tensor
from transformers import AutoConfig, AutoModel


class SimLBRCritic(nn.Module):
    def __init__(self, backbone_path, lightning_state, size=256):
        super().__init__()
        config = AutoConfig.from_pretrained(Path(backbone_path), local_files_only=True)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(256, 1),
        )
        self.size = int(size)
        state = {}
        for name, value in lightning_state.items():
            if name.startswith('backbone.model.'):
                state['model.' + name.removeprefix('backbone.model.')] = value
            elif name.startswith('classifier.'):
                state[name] = value
        self.load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)

    def forward(self, normalized_images):
        def features(images):
            return self.model(pixel_values=images).last_hidden_state[:, 0]
        with torch.autocast(normalized_images.device.type, dtype=torch.bfloat16,
                            enabled=normalized_images.is_cuda):
            if torch.is_grad_enabled() and normalized_images.requires_grad:
                cls = checkpoint(features, normalized_images, use_reentrant=False)
            else:
                cls = features(normalized_images)
            return self.classifier(cls).flatten().float()

    @staticmethod
    def normalize(images):
        mean = images.new_tensor((.430, .411, .296))[None, :, None, None]
        std = images.new_tensor((.213, .156, .143))[None, :, None, None]
        return (images - mean) / std

    def score_images(self, rgb_images):
        resized = F.interpolate(rgb_images, (self.size, self.size), mode='bilinear',
                                align_corners=False, antialias=True)
        return self(self.normalize(resized))

    def score_pil(self, rgb_image: Image.Image):
        tensor = to_tensor(rgb_image).unsqueeze(0).to(next(self.parameters()).device)
        return self.score_images(tensor)
