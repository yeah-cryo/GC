"""Differentiable low-noise SD reconstruction guided by a trainable RGB critic."""
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from aigi_detection.losses.losses import critic_logits
from .classifier_guidance import clean_prediction


def low_noise_timesteps(sd, maximum):
    sd.scheduler.set_timesteps(sd.steps, device=sd.device)
    timesteps = [t for t in sd.scheduler.timesteps if int(t) <= int(maximum)]
    if not timesteps:
        raise ValueError(f'No DDIM timestep is at or below {maximum}.')
    return timesteps


def encode_image(sd, image):
    posterior = sd.vae.encode((image * 2 - 1).to(sd.dtype)).latent_dist
    return posterior.mode().float() * sd.vae.config.scaling_factor


def decode_latent(sd, latent, checkpointing):
    def decode(value):
        decoded = sd.vae.decode((value / sd.vae.config.scaling_factor).to(sd.dtype)).sample
        return decoded.float() / 2 + 0.5
    if checkpointing and torch.is_grad_enabled() and latent.requires_grad:
        return checkpoint(decode, latent, use_reentrant=False).clamp(0, 1)
    return decode(latent).clamp(0, 1)


def predict_noise(sd, latent, timestep, conditioning, checkpointing):
    def predict(value, context, timestep=timestep):
        model_input = sd.scheduler.scale_model_input(value.to(sd.dtype), timestep)
        return sd.unet(model_input, timestep, encoder_hidden_states=context).sample.float()
    if checkpointing and torch.is_grad_enabled() and latent.requires_grad:
        return checkpoint(predict, latent, conditioning, use_reentrant=False)
    return predict(latent, conditioning)


def differentiable_guided_epsilon(epsilon, gradient, alpha, strength, rms_clip):
    if strength < 0 or rms_clip <= 0:
        raise ValueError('Guidance strength must be nonnegative and RMS cap positive.')
    if not math.isfinite(strength) or not math.isfinite(rms_clip):
        raise ValueError('Guidance strength and RMS cap must be finite.')
    gradient = gradient.float()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite reconstruction-guidance gradient.')
    rms = gradient.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    clipped = gradient * (rms_clip / rms.clamp_min(1e-12)).clamp(max=1)
    return epsilon.float() + strength * (1 - alpha).sqrt() * clipped, rms.mean()


def _unguided_step(sd, latent, timestep, conditioning, checkpointing=False):
    epsilon = predict_noise(sd, latent, timestep, conditioning, checkpointing)
    return sd.scheduler.step(epsilon, timestep, latent.float(), eta=0.0).prev_sample


def reconstruct_with_real_guidance(sd, critic, image, conditioning, noise, maximum_timestep=200,
                                   differentiable_steps=2, strength=20.0, rms_clip=0.1,
                                   checkpointing=True, second_order=True):
    """Return matched unguided/guided reconstructions from one noised image latent.

    Early reverse steps and the unguided reference are detached. The final
    ``differentiable_steps`` retain second-order gradients through classifier guidance
    when ``second_order`` is true. Disable it when measuring a frozen critic to release
    each step's graph immediately and substantially reduce memory use.
    """
    if differentiable_steps < 1:
        raise ValueError('differentiable_steps must be positive.')
    if sd.scheduler.config.prediction_type != 'epsilon' or sd.scheduler.config.clip_sample:
        raise ValueError('Reconstruction requires epsilon prediction and unclipped DDIM x0.')
    conditioning = conditioning.detach().to(device=sd.device, dtype=sd.dtype)
    timesteps = low_noise_timesteps(sd, maximum_timestep)
    if differentiable_steps > len(timesteps):
        raise ValueError('differentiable_steps exceeds available low-noise DDIM steps.')
    with torch.no_grad():
        source_latent = encode_image(sd, image)
        latent = sd.scheduler.add_noise(source_latent, noise.float(), timesteps[0])
        for timestep in timesteps[:-differentiable_steps]:
            latent = _unguided_step(sd, latent, timestep, conditioning)
        shared_latent = latent.detach()
        baseline_latent = shared_latent
        for timestep in timesteps[-differentiable_steps:]:
            baseline_latent = _unguided_step(sd, baseline_latent, timestep, conditioning)
        baseline_image = decode_latent(sd, baseline_latent, checkpointing=False).detach()

    guided_latent = shared_latent.detach().requires_grad_(True)
    guidance_rms = []
    for timestep in timesteps[-differentiable_steps:]:
        epsilon = predict_noise(sd, guided_latent, timestep, conditioning, checkpointing)
        alpha = sd.scheduler.alphas_cumprod[int(timestep)].to(sd.device, torch.float32)
        predicted_x0 = clean_prediction(guided_latent, epsilon, alpha)
        estimate = decode_latent(sd, predicted_x0, checkpointing)
        logits = critic_logits(critic, estimate)
        real_target = torch.zeros_like(logits)
        guidance_loss = F.binary_cross_entropy_with_logits(logits, real_target)
        gradient = torch.autograd.grad(
            guidance_loss, guided_latent, create_graph=second_order,
            retain_graph=second_order)[0]
        corrected, rms = differentiable_guided_epsilon(
            epsilon, gradient, alpha, strength, rms_clip)
        guided_latent = sd.scheduler.step(
            corrected, timestep, guided_latent.float(), eta=0.0).prev_sample
        if not second_order:
            guided_latent = guided_latent.detach().requires_grad_(True)
        guidance_rms.append(rms.detach())
    if second_order:
        guided_image = decode_latent(sd, guided_latent, checkpointing)
    else:
        with torch.no_grad():
            guided_image = decode_latent(sd, guided_latent, checkpointing=False).detach()
        guided_latent = guided_latent.detach()
    return {
        'source_latent': source_latent.detach(),
        'baseline_latent': baseline_latent.detach(),
        'guided_latent': guided_latent,
        'baseline_image': baseline_image,
        'guided_image': guided_image,
        'timesteps': [int(t) for t in timesteps],
        'guided_timesteps': [int(t) for t in timesteps[-differentiable_steps:]],
        'mean_guidance_gradient_rms': torch.stack(guidance_rms).mean(),
    }


def reconstruction_components(result, perceptual, latent_std):
    std = result['guided_latent'].new_tensor(latent_std)[None, :, None, None]
    latent = ((result['guided_latent'] - result['baseline_latent']) / std).square().mean()
    pixel = (result['guided_image'] - result['baseline_image']).abs().mean()
    lpips = perceptual(result['guided_image'].float(), result['baseline_image'].float()).mean()
    return {'latent': latent, 'pixel': pixel, 'lpips': lpips}


def reconstruction_objective(real, fake, weights, margin):
    def score(parts):
        return sum(float(weights[name]) * parts[name] for name in ('latent', 'pixel', 'lpips'))
    real_score, fake_score = score(real), score(fake)
    ranking = F.relu(real_score - fake_score + float(margin))
    return real_score, fake_score, ranking
