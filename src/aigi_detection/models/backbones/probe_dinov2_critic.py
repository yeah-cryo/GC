"""Differentiable adapter for PROBE's DINOv2-L/14-with-registers linear critic."""
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_tensor
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from aigi_detection.losses.losses import normalize


class PROBEDINOv2Critic(nn.Module):
    """PROBE checkpoint with its official 336-pixel sliding-crop preprocessing."""

    def __init__(self, state, crop_size=336):
        super().__init__()
        config = Dinov2WithRegistersConfig(
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
            patch_size=14, image_size=518, num_register_tokens=4,
            mlp_ratio=4, qkv_bias=True, use_swiglu_ffn=False,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-6, layerscale_value=1.0,
        )
        self.backbone = Dinov2WithRegistersModel(config)
        self.fc = nn.Linear(1024, 1)
        self.crop_size = int(crop_size)
        self.load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)

    def forward(self, normalized_patches):
        def features(images):
            return self.backbone(pixel_values=images).last_hidden_state[:, 0]
        with torch.autocast(normalized_patches.device.type, dtype=torch.bfloat16,
                            enabled=normalized_patches.is_cuda):
            if torch.is_grad_enabled() and normalized_patches.requires_grad:
                cls = checkpoint(features, normalized_patches, use_reentrant=False)
            else:
                cls = features(normalized_patches)
            return self.fc(cls).flatten().float()

    def _patches(self, rgb_images):
        size = self.crop_size
        height, width = rgb_images.shape[-2:]
        if height < size or width < size:
            rgb_images = F.pad(rgb_images, (0, max(0, size - width),
                                             0, max(0, size - height)), mode='replicate')
        # Match PROBE's Tensor.unfold(size, stride=size), including its top-left
        # single patch for a 512x512 SD image and crop_size=336.
        patches = rgb_images.unfold(2, size, size).unfold(3, size, size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.flatten(0, 2)

    def score_images(self, rgb_images):
        batch = rgb_images.shape[0]
        patches = self._patches(rgb_images)
        logits = self(normalize(patches))
        patches_per_image = logits.numel() // batch
        return logits.reshape(batch, patches_per_image).mean(1)

    def score_pil(self, rgb_image):
        tensor = to_tensor(rgb_image).unsqueeze(0).to(next(self.parameters()).device)
        return self.score_images(tensor)
