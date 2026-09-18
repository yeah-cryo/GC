from types import SimpleNamespace

import torch
from diffusers import DDIMScheduler
from torch import nn
from torch.nn import functional as F

from aigi_detection.models.modules.differentiable_sd import DifferentiableSD
from aigi_detection.models.modules.guided_reconstruction import (
    reconstruct_with_real_guidance, reconstruction_components, reconstruction_objective)


class Posterior:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value


class VAE(nn.Module):
    config = SimpleNamespace(scaling_factor=1.)

    def encode(self, image):
        latent = F.avg_pool2d(image, 8)
        latent = torch.cat((latent, latent.mean(1, keepdim=True)), dim=1)
        return SimpleNamespace(latent_dist=Posterior(latent))

    def decode(self, latent):
        image = F.interpolate(latent[:, :3], scale_factor=8, mode='bilinear', align_corners=False)
        return SimpleNamespace(sample=image)


class UNet(nn.Module):
    config = SimpleNamespace(in_channels=4)

    def forward(self, latent, timestep, encoder_hidden_states):
        return SimpleNamespace(sample=0.05 * latent + 0.001 * encoder_hidden_states.mean())


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        return (self.scale * image.mean((1, 2, 3)))[:, None]


class Perceptual(nn.Module):
    def forward(self, first, second):
        return (first - second).abs().mean((1, 2, 3))


def make_sd():
    sd = DifferentiableSD.__new__(DifferentiableSD)
    sd.device, sd.dtype = torch.device('cpu'), torch.float32
    sd.size, sd.steps, sd.guidance = 32, 5, 1.0
    sd.unet, sd.vae = UNet(), VAE()
    sd.scheduler = DDIMScheduler(num_train_timesteps=25, clip_sample=False)
    return sd


def test_second_order_reconstruction_gradient_reaches_critic():
    sd, critic = make_sd(), Critic()
    image = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    conditioning = torch.zeros(1, 4, 8)
    noise = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(4))
    result = reconstruct_with_real_guidance(
        sd, critic, image, conditioning, noise, maximum_timestep=10,
        differentiable_steps=2, strength=2.0, rms_clip=0.1)
    parts = reconstruction_components(result, Perceptual(), [1, 1, 1, 1])
    sum(parts.values()).backward()
    assert critic.scale.grad is not None
    assert torch.isfinite(critic.scale.grad) and critic.scale.grad.abs() > 0
    assert result['guided_latent'].requires_grad
    assert not result['baseline_latent'].requires_grad
    assert len(result['guided_timesteps']) == 2


def test_frozen_critic_reconstruction_releases_graph():
    sd, critic = make_sd(), Critic().requires_grad_(False)
    image = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    conditioning = torch.zeros(1, 4, 8)
    noise = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(7))
    result = reconstruct_with_real_guidance(
        sd, critic, image, conditioning, noise, maximum_timestep=10,
        differentiable_steps=2, strength=2.0, rms_clip=0.1,
        checkpointing=False, second_order=False)
    assert not result['guided_latent'].requires_grad
    assert not result['guided_image'].requires_grad
    assert not torch.equal(result['guided_latent'], result['baseline_latent'])


def test_ranking_loss_is_bounded_by_margin():
    real = {'latent': torch.tensor(.1), 'pixel': torch.tensor(.2), 'lpips': torch.tensor(.3)}
    fake = {'latent': torch.tensor(.4), 'pixel': torch.tensor(.5), 'lpips': torch.tensor(.6)}
    weights = {'latent': 1., 'pixel': 1., 'lpips': 1.}
    real_score, fake_score, ranking = reconstruction_objective(real, fake, weights, margin=.2)
    torch.testing.assert_close(real_score, torch.tensor(.6))
    torch.testing.assert_close(fake_score, torch.tensor(1.5))
    assert ranking == 0
    _, _, active = reconstruction_objective(real, real, weights, margin=.2)
    torch.testing.assert_close(active, torch.tensor(.2))
