"""Differentiable adapter for the released EFFORT CLIP ViT-L/14 detector."""
from collections import OrderedDict

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_tensor
from transformers import CLIPVisionConfig, CLIPVisionModel


def _effective_state(checkpoint_state):
    """Merge EFFORT's rank-one residual attention factors into linear weights."""
    state = {name.removeprefix('module.'): value for name, value in checkpoint_state.items()}
    backbone = OrderedDict()
    prefix = 'backbone.'
    for name, value in state.items():
        if not name.startswith(prefix):
            continue
        local_name = name.removeprefix(prefix)
        if local_name.endswith(('.S_residual', '.U_residual', '.V_residual')):
            continue
        if local_name.endswith('.weight_main'):
            base = local_name.removesuffix('.weight_main')
            singular = state[f'{prefix}{base}.S_residual']
            left = state[f'{prefix}{base}.U_residual']
            right = state[f'{prefix}{base}.V_residual']
            backbone[f'{base}.weight'] = value + (left * singular.unsqueeze(0)) @ right
        else:
            backbone[local_name] = value
    head = {'weight': state['head.weight'], 'bias': state['head.bias']}
    return backbone, head


class EffortCritic(nn.Module):
    def __init__(self, checkpoint_state, image_size=224):
        super().__init__()
        config = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            hidden_act='quick_gelu',
        )
        # Current Transformers exposes the vision transformer directly here; its
        # state names match EFFORT's checkpoint after removing ``backbone.``.
        self.backbone = CLIPVisionModel(config)
        self.head = nn.Linear(1024, 2)
        backbone_state, head_state = _effective_state(checkpoint_state)
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.head.load_state_dict(head_state, strict=True)
        self.image_size = int(image_size)
        self.eval().requires_grad_(False)

    @staticmethod
    def normalize(images):
        mean = images.new_tensor((.48145466, .4578275, .40821073))[None, :, None, None]
        std = images.new_tensor((.26862954, .26130258, .27577711))[None, :, None, None]
        return (images - mean) / std

    def forward(self, normalized_images):
        def features(images):
            return self.backbone(images).pooler_output
        with torch.autocast(normalized_images.device.type, dtype=torch.bfloat16,
                            enabled=normalized_images.is_cuda):
            if torch.is_grad_enabled() and normalized_images.requires_grad:
                pooled = checkpoint(features, normalized_images, use_reentrant=False)
            else:
                pooled = features(normalized_images)
            logits = self.head(pooled)
        # sigmoid(class-1 logit minus class-0 logit) equals softmax probability of fake.
        return (logits[:, 1] - logits[:, 0]).float()

    def score_images(self, rgb_images):
        resized = F.interpolate(rgb_images, (self.image_size, self.image_size),
                                mode='bilinear', align_corners=False, antialias=False)
        return self(self.normalize(resized))

    def score_pil(self, rgb_image: Image.Image):
        tensor = to_tensor(rgb_image).unsqueeze(0).to(next(self.parameters()).device)
        return self.score_images(tensor)
