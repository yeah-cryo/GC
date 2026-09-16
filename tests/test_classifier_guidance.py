from types import SimpleNamespace

import torch
from torch import nn
from diffusers import DDIMScheduler

from aigi_detection.models.modules.classifier_guidance import clean_prediction, guided_epsilon, sample_guided
from aigi_detection.models.modules.differentiable_sd import DifferentiableSD
from aigi_detection.losses.losses import critic_logits


def test_guidance_sign_lowers_real_target_loss():
    alpha = torch.tensor(0.5)
    x = torch.ones(1, 4, 2, 2, requires_grad=True)
    epsilon = x * 0.1
    x0 = clean_prediction(x, epsilon, alpha)
    loss = torch.nn.functional.softplus(x0.mean())  # BCE(fake logit, real=0)
    gradient = torch.autograd.grad(loss, x)[0]
    corrected, raw, clipped = guided_epsilon(epsilon.detach(), gradient, alpha, 1., 0.1)
    assert torch.nn.functional.softplus(clean_prediction(x.detach(), corrected, alpha).mean()) < loss
    assert clipped <= raw and clipped <= 0.100001


class UNet(nn.Module):
    config = SimpleNamespace(in_channels=4)
    def forward(self, x, t, encoder_hidden_states):
        return SimpleNamespace(sample=0.01 * x + encoder_hidden_states.mean() * 0.01)


class VAE(nn.Module):
    config = SimpleNamespace(scaling_factor=1.)
    def decode(self, x):
        return SimpleNamespace(sample=x[:, :3] * 0.1)


class Critic(nn.Module):
    def forward(self, x):
        return x[:, :, 110:114, 110:114].mean((1, 2, 3))[:, None]


def make_sd():
    sd = DifferentiableSD.__new__(DifferentiableSD)
    sd.device, sd.dtype = torch.device('cpu'), torch.float32
    sd.size, sd.steps, sd.guidance = 32, 4, 7.5
    sd.unet, sd.vae = UNet(), VAE()
    sd.scheduler = DDIMScheduler(num_train_timesteps=20, clip_sample=False)
    return sd


def test_zero_guidance_and_disabled_window_match_standard_sampler():
    sd, critic = make_sd(), Critic()
    context = torch.zeros(1, 4, 8)
    with torch.no_grad():
        expected = sd.generate(context, context, 42)
    for strength, maximum in [(0, 200), (5, -1)]:
        image, logs, _ = sample_guided(sd, critic, context, context, 42, strength, maximum)
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        assert not any(row['guided'] for row in logs)


def test_only_low_noise_steps_are_guided_towards_real():
    sd, critic = make_sd(), Critic()
    context = torch.zeros(1, 4, 8)
    baseline, _, _ = sample_guided(sd, critic, context, context, 42, 0)
    guided, logs, images = sample_guided(sd, critic, context, context, 42, 5, 6, capture_estimates=True)
    assert [row['timestep'] for row in logs if row['guided']] == [5, 0]
    assert len(images) == 2 and not guided.requires_grad
    assert all(row['gradient_rms'] > 0 for row in logs if row['guided'])
    assert critic_logits(critic, guided) < critic_logits(critic, baseline)
