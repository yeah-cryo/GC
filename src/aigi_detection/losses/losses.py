import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


def normalize(image):
    mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (image - mean) / std


def critic_logits(critic, image, crop_size=224):
    if hasattr(critic, 'score_images'):
        return critic.score_images(image)
    height, width = image.shape[-2:]
    if min(height, width) < crop_size:
        dy, dx = max(0, crop_size - height), max(0, crop_size - width)
        image = F.pad(image, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        height, width = image.shape[-2:]
    positions = [(0, 0), (0, width - crop_size), (height - crop_size, 0),
                 (height - crop_size, width - crop_size), (round((height - crop_size) / 2), round((width - crop_size) / 2))]
    logits = [critic(normalize(image[..., y:y + crop_size, x:x + crop_size])).flatten().float() for y, x in positions]
    return torch.stack(logits).mean(0)


class PerceptualFeatures(nn.Module):
    """Independent frozen ImageNet ResNet-50 layer2/layer3 channel-normalized features."""
    def __init__(self, weights):
        super().__init__()
        model = resnet50(weights=None)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
        self.layer2, self.layer3 = model.layer2, model.layer3
        self.eval().requires_grad_(False)

    def forward(self, images):
        images = normalize(F.interpolate(images, (224, 224), mode='bilinear', align_corners=False, antialias=True))
        layer2 = self.layer2(self.stem(images))
        layer3 = self.layer3(layer2)
        return tuple(F.normalize(features, dim=1) for features in (layer2, layer3))


def perceptual_loss(generated_features, reference_features):
    return torch.stack([(a - b).square().mean() for a, b in zip(generated_features, reference_features)]).mean()
