"""Adapter for GGGT's frozen original DINOv3-L/16 + MLP critic."""
import importlib.util
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_tensor

from aigi_detection.losses.losses import normalize


class DINOv3Critic(nn.Module):
    def __init__(self, implementation, backbone, head_state):
        super().__init__()
        spec = importlib.util.spec_from_file_location('gggt_frozen_dinov3', implementation)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.DINOv3ViTModel.from_pretrained(str(Path(backbone) / 'model.safetensors'))
        self.fc1 = nn.Linear(1024, 768)
        self.fc2 = nn.Linear(768, 1)
        self.fc1.load_state_dict({key.removeprefix('fc1.'): value for key, value in head_state.items() if key.startswith('fc1.')})
        self.fc2.load_state_dict({key.removeprefix('fc2.'): value for key, value in head_state.items() if key.startswith('fc2.')})
        self.eval().requires_grad_(False)

    def forward(self, normalized_images):
        # Match GGGT's evaluation precision, including the MLP autocast.
        with torch.autocast(normalized_images.device.type, dtype=torch.float16, enabled=normalized_images.is_cuda):
            def features(images):
                return self.model(images)['cls_token']
            if torch.is_grad_enabled() and normalized_images.requires_grad:
                cls = checkpoint(features, normalized_images, use_reentrant=False)
            else:
                cls = features(normalized_images)
            embedding = F.normalize(cls.float(), p=2, dim=-1)
            return self.fc2(F.gelu(self.fc1(embedding))).flatten().float()

    def score_images(self, rgb_images):
        resized = F.interpolate(rgb_images, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        return self(normalize(resized))

    def score_pil(self, rgb_image):
        # Exact saved-image preprocessing from GGGT datasets.transforms.val_transform.
        tensor = to_tensor(rgb_image.resize((224, 224), Image.Resampling.BILINEAR)).unsqueeze(0)
        return self(normalize(tensor.to(next(self.parameters()).device)))
