"""Full-gradient DDIM sampling; no inference pipeline or latent detachment."""
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class DifferentiableSD:
    def __init__(self, path, device='cuda', dtype=torch.bfloat16, steps=35, guidance=7.5, size=512):
        path = Path(path)
        self.device, self.dtype = torch.device(device), dtype
        self.steps, self.guidance, self.size = steps, guidance, size
        self.tokenizer = CLIPTokenizer.from_pretrained(path / 'tokenizer', local_files_only=True)
        self.text_encoder = CLIPTextModel.from_pretrained(path / 'text_encoder', local_files_only=True).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(path / 'unet', local_files_only=True, torch_dtype=dtype).to(device)
        self.vae = AutoencoderKL.from_pretrained(path / 'vae', local_files_only=True, torch_dtype=dtype).to(device)
        self.scheduler = DDIMScheduler.from_pretrained(path / 'scheduler', local_files_only=True, clip_sample=False)
        for model in self.models():
            model.eval().requires_grad_(False)

    def models(self):
        return (self.text_encoder, self.unet, self.vae)

    def generate(self, conditioning, unconditional, seed, checkpointing=True, trace=None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn((1, self.unet.config.in_channels, self.size // 8, self.size // 8),
                              device=self.device, dtype=self.dtype, generator=generator)
        self.scheduler.set_timesteps(self.steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma
        context = torch.cat([unconditional, conditioning]).to(self.dtype)
        def trace_tensor(name, tensor):
            if trace is not None and tensor.requires_grad:
                tensor.register_hook(lambda gradient: trace.__setitem__(name, float(gradient.float().norm())))
        trace_tensor('conditioning_gradient_norm', conditioning)
        for timestep in self.scheduler.timesteps:
            inputs = self.scheduler.scale_model_input(torch.cat([latents, latents]), timestep)
            # Bind timestep now: backward recomputation must never capture a later loop value.
            def predict(x, context, timestep=timestep):
                return self.unet(x, timestep, encoder_hidden_states=context).sample
            if checkpointing and torch.is_grad_enabled():
                noise = checkpoint(predict, inputs, context, use_reentrant=False)
            else:
                noise = predict(inputs, context)
            uncond, cond = noise.chunk(2)
            noise = uncond + self.guidance * (cond - uncond)
            latents = self.scheduler.step(noise, timestep, latents, eta=0.0).prev_sample
        trace_tensor('final_latent_gradient_norm', latents)
        def decode(z):
            return self.vae.decode(z / self.vae.config.scaling_factor).sample
        if checkpointing and torch.is_grad_enabled():
            decoded = checkpoint(decode, latents, use_reentrant=False)
        else:
            decoded = decode(latents)
        trace_tensor('decoded_image_gradient_norm', decoded)
        return (decoded.float() / 2 + 0.5).clamp(0, 1)
