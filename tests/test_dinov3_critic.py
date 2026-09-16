import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from aigi_detection.models.backbones.dinov3_critic import DINOv3Critic
from aigi_detection.losses.losses import critic_logits


class TinyBackbone(nn.Module):
    def forward(self, x):
        assert x.shape[-2:] == (224, 224)
        return {'cls_token': torch.cat([x.mean((2, 3)), x.square().mean((2, 3))], dim=1)}


def critic():
    result = DINOv3Critic.__new__(DINOv3Critic)
    nn.Module.__init__(result)
    result.model = TinyBackbone()
    result.fc1, result.fc2 = nn.Linear(6, 4), nn.Linear(4, 1)
    return result.eval().requires_grad_(False)


def test_dinov3_dispatch_preserves_input_gradients():
    model = critic()
    image = torch.rand(1, 3, 128, 256, requires_grad=True)
    output = critic_logits(model, image)
    output.sum().backward()
    assert image.grad is not None and image.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())


def test_pil_scoring_matches_gggt_preprocessing():
    model = critic()
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (317, 512, 3), dtype=np.uint8))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    torch.testing.assert_close(model.score_pil(image), model(transform(image).unsqueeze(0)), rtol=0, atol=0)
