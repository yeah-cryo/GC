"""Low-noise, predicted-x0 classifier guidance with a clean-image critic."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from aigi_detection.losses.losses import critic_logits


def clean_prediction(latents, epsilon, alpha):
    return (latents.float() - (1 - alpha).sqrt() * epsilon.float()) / alpha.sqrt()


def guided_epsilon(epsilon, loss_gradient, alpha, strength, rms_clip):
    """Positive BCE gradient enters epsilon; the DDIM update then descends BCE."""
    gradient = loss_gradient.float()
    rms = gradient.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite classifier guidance gradient')
    clipped = gradient * (rms_clip / rms.clamp_min(1e-12)).clamp(max=1)
    epsilon = epsilon.float() + strength * (1 - alpha).sqrt() * clipped
    return epsilon, rms.mean().item(), clipped.square().mean().sqrt().item()


def sample_guided(sd, critic, conditioning, unconditional, seed, strength=5.0,
                  max_timestep=200, gradient_rms_clip=0.1, capture_estimates=False):
    if strength < 0 or gradient_rms_clip <= 0:
        raise ValueError('Guidance strength must be nonnegative and gradient RMS cap positive.')
    if sd.scheduler.config.prediction_type != 'epsilon' or sd.scheduler.config.clip_sample:
        raise ValueError('This SD1.4 sampler requires epsilon prediction and unclipped DDIM x0.')
    context = torch.cat((unconditional, conditioning)).detach().to(sd.dtype)
    rng = torch.Generator(device=sd.device).manual_seed(seed)
    latents = torch.randn((1, sd.unet.config.in_channels, sd.size // 8, sd.size // 8),
                          generator=rng, device=sd.device, dtype=sd.dtype) * sd.scheduler.init_noise_sigma
    sd.scheduler.set_timesteps(sd.steps, device=sd.device)
    logs, estimates = [], []
    for timestep in sd.scheduler.timesteps:
        t = int(timestep)
        active = strength > 0 and t <= max_timestep
        def predict(x, timestep=timestep):
            inputs = sd.scheduler.scale_model_input(torch.cat((x, x)), timestep)
            prediction = sd.unet(inputs, timestep, encoder_hidden_states=context).sample
            unconditional_noise, conditional_noise = prediction.chunk(2)
            return unconditional_noise + sd.guidance * (conditional_noise - unconditional_noise)
        if active:
            # Each step creates its own graph. No graph connects separate sampling steps.
            with torch.enable_grad():
                x = latents.detach().requires_grad_(True)
                epsilon = checkpoint(predict, x, use_reentrant=False)
                alpha = sd.scheduler.alphas_cumprod[t].to(device=sd.device, dtype=torch.float32)
                predicted_x0 = clean_prediction(x, epsilon, alpha)
                def decode(z):
                    return sd.vae.decode((z / sd.vae.config.scaling_factor).to(sd.dtype)).sample
                decoded = checkpoint(decode, predicted_x0, use_reentrant=False)
                estimate = (decoded.float() / 2 + 0.5).clamp(0, 1)
                logits = critic_logits(critic, estimate)
                loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
                gradient = torch.autograd.grad(loss, x)[0]
            with torch.no_grad():
                corrected, raw_rms, clipped_rms = guided_epsilon(epsilon.detach(), gradient, alpha, strength, gradient_rms_clip)
                latents = sd.scheduler.step(corrected, timestep, x.detach().float(), eta=0.0).prev_sample.to(sd.dtype)
                logs.append({'timestep': t, 'guided': True, 'estimated_x0_fake_probability': logits.sigmoid().item(),
                             'estimated_x0_bce': loss.item(), 'gradient_rms': raw_rms, 'clipped_gradient_rms': clipped_rms})
                if capture_estimates:
                    estimates.append(estimate.detach().cpu())
            del x, epsilon, predicted_x0, decoded, estimate, logits, loss, gradient, corrected
        else:
            with torch.no_grad():
                epsilon = predict(latents)
                latents = sd.scheduler.step(epsilon, timestep, latents, eta=0.0).prev_sample
            logs.append({'timestep': t, 'guided': False})
        latents = latents.detach()
    with torch.no_grad():
        image = (sd.vae.decode(latents / sd.vae.config.scaling_factor).sample.float() / 2 + 0.5).clamp(0, 1)
    return image, logs, estimates
