"""Differentiable adapter for the released FreqNet detector."""
import importlib.util
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_tensor


def _load_freqnet(repository):
    source = Path(repository) / 'networks' / 'freqnet.py'
    spec = importlib.util.spec_from_file_location('_released_freqnet', source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import FreqNet from {source}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.freqnet


class FreqNetCritic(nn.Module):
    def __init__(self, repository, checkpoint_state):
        super().__init__()
        # The released constructor creates four parameters directly on CUDA.
        # This adapter is used only by the CUDA guidance workflow.
        if not torch.cuda.is_available():
            raise RuntimeError('The released FreqNet constructor requires CUDA.')
        self.model = _load_freqnet(repository)()
        self.model.load_state_dict(checkpoint_state, strict=True)
        self.model.eval().requires_grad_(False)

    @staticmethod
    def normalize(images):
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return (images - mean) / std

    def forward(self, normalized_images):
        def classify(images):
            return self.model(images).flatten()

        if torch.is_grad_enabled() and normalized_images.requires_grad:
            logits = checkpoint(classify, normalized_images, use_reentrant=False)
        else:
            logits = classify(normalized_images)
        return logits.float()

    def score_images(self, rgb_images):
        # FreqNet's GANGen inference path keeps the original resolution and
        # applies only ImageNet normalization.
        return self(self.normalize(rgb_images.float()))

    def score_pil(self, rgb_image: Image.Image):
        tensor = to_tensor(rgb_image).unsqueeze(0).to(next(self.parameters()).device)
        return self.score_images(tensor)
