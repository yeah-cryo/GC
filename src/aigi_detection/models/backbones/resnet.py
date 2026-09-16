from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50


def build_critic(pretrained=None):
    """Load local ImageNet weights before initializing the binary logit head."""
    model = resnet50(weights=None)
    if pretrained is not None:
        path = Path(pretrained)
        if path.is_dir():
            candidates = sorted(p for p in path.iterdir() if p.suffix in {'.pt', '.pth', '.bin'})
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one PyTorch checkpoint in {path}: {candidates}')
            path = candidates[0]
        state = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('state_dict', 'model'):
            if key in state:
                state = state[key]
                break
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.requires_grad_(True)
    return model
