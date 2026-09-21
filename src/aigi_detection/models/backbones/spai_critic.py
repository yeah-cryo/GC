"""Differentiable adapter for the released SPAI spectral detector."""
from pathlib import Path
import sys
import types

import packaging
import packaging.version
import pkg_resources
import torch
from PIL import Image
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_tensor


def _load_spai_modules(repository):
    """Import SPAI without pulling in its logging-only seaborn dependency."""
    repository = str(Path(repository).resolve())
    if repository not in sys.path:
        sys.path.insert(0, repository)

    # The OpenAI CLIP version imported by SPAI expects packaging to be exposed
    # through pkg_resources. Current setuptools no longer exposes that alias.
    pkg_resources.packaging = packaging

    import spai
    if 'spai.utils' not in sys.modules:
        utilities = types.ModuleType('spai.utils')
        utilities.save_image_with_attention_overlay = lambda *args, **kwargs: None
        sys.modules['spai.utils'] = utilities

    from spai.config import get_custom_config
    from spai.models import build_cls_model
    return get_custom_config, build_cls_model


class SPAICritic(nn.Module):
    def __init__(self, repository, checkpoint_state, config_path=None):
        super().__init__()
        # PyTorch 2.10 may route the attention backward pass through an
        # optional Triton bmm kernel. Keep eager bmm here so this experiment
        # does not require a local C compiler/Python development headers.
        try:
            import torch._native.ops.bmm_outer_product  # noqa: F401
            from torch._native.registry import deregister_op_overrides
            deregister_op_overrides(disable_op_symbols='bmm')
        except (ImportError, AttributeError):
            pass
        repository = Path(repository)
        config_path = Path(config_path or repository / 'configs' / 'spai.yaml')
        get_custom_config, build_cls_model = _load_spai_modules(repository)
        config = get_custom_config(str(config_path))
        self.model = build_cls_model(config)
        state = checkpoint_state['model'] if 'model' in checkpoint_state else checkpoint_state
        self.model.load_state_dict(state, strict=True)

        # SPAI normally wraps its frozen backbone in no_grad. Parameters remain
        # frozen here, but this flag must be disabled for an image gradient.
        self.model.unfreeze_backbone()
        self.model.eval().requires_grad_(False)

    def forward(self, rgb_images):
        def classify(images):
            return self.model(images).flatten()

        with torch.autocast(rgb_images.device.type, dtype=torch.bfloat16,
                            enabled=rgb_images.is_cuda):
            if torch.is_grad_enabled() and rgb_images.requires_grad:
                logits = checkpoint(classify, rgb_images, use_reentrant=False)
            else:
                logits = classify(rgb_images)
        return logits.float()

    def score_images(self, rgb_images):
        return self(rgb_images)

    def score_pil(self, rgb_image: Image.Image):
        tensor = to_tensor(rgb_image).unsqueeze(0).to(next(self.parameters()).device)
        return self.score_images(tensor)
