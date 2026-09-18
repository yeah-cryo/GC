"""Train a ResNet critic through low-noise SD1.4 reconstruction guidance."""
import argparse
import hashlib
import json
import math
import os
import time
import uuid
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from aigi_detection.datasets import ReconstructionPairDataset, official_records
from aigi_detection.engine import atomic_save, restore_rng, rng_state, seed_everything, write_json
from aigi_detection.losses.losses import critic_logits
from aigi_detection.models.backbones import build_critic
from aigi_detection.models.modules.differentiable_sd import DifferentiableSD
from aigi_detection.models.modules.guided_reconstruction import (
    reconstruct_with_real_guidance, reconstruction_components, reconstruction_objective)
from aigi_detection.models.modules.soft_prompt import encode_plain


LABELS = {'nature': 0, 'ai': 1}


def manifest(records):
    text = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records)
    return text, hashlib.sha256(text.encode()).hexdigest()


def save_state(path, critic, optimizer, scheduler, cursor, optimizer_steps, config,
               manifest_hash, history):
    atomic_save({
        'model': critic.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'cursor': cursor,
        'completed_pairs': cursor, 'optimizer_steps': optimizer_steps,
        'configuration': config, 'pair_manifest_sha256': manifest_hash,
        'label_mapping': LABELS, 'history': history,
        'metrics': history[-1] if history else None, 'rng_state': rng_state(),
        'architecture': 'resnet50', 'output_semantics': 'one logit; sigmoid = P(fake)',
    }, path)


def freeze_batchnorm(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def make_noise(sd, seed):
    generator = torch.Generator(device=sd.device).manual_seed(int(seed))
    return torch.randn((1, sd.unet.config.in_channels, sd.size // 8, sd.size // 8),
                       device=sd.device, dtype=torch.float32, generator=generator)


def pair_objective(sd, critic, perceptual, conditioning, real, fake, config, seed):
    noise = make_noise(sd, seed)
    kwargs = {
        'conditioning': conditioning, 'noise': noise,
        'maximum_timestep': config['maximum_noise_timestep'],
        'differentiable_steps': config['differentiable_guidance_steps'],
        'strength': config['guidance_strength'], 'rms_clip': config['gradient_rms_clip'],
        'checkpointing': True,
    }
    real_result = reconstruct_with_real_guidance(sd, critic, real, **kwargs)
    fake_result = reconstruct_with_real_guidance(sd, critic, fake, **kwargs)
    real_parts = reconstruction_components(
        real_result, perceptual, config['latent_channel_std'])
    fake_parts = reconstruction_components(
        fake_result, perceptual, config['latent_channel_std'])
    real_score, fake_score, ranking = reconstruction_objective(
        real_parts, fake_parts, config['reconstruction_weights'], config['ranking_margin'])
    logits = critic_logits(critic, torch.cat((real, fake)))
    labels = logits.new_tensor((0., 1.))
    classification = F.binary_cross_entropy_with_logits(logits, labels)
    loss = (config['classification_weight'] * classification
            + config['real_preservation_weight'] * real_score
            + config['ranking_weight'] * ranking)
    metrics = {
        'loss': loss.detach().item(), 'classification_loss': classification.detach().item(),
        'real_reconstruction': real_score.detach().item(),
        'fake_reconstruction': fake_score.detach().item(),
        'ranking_loss': ranking.detach().item(),
        'real_fake_probability': logits[0].detach().sigmoid().item(),
        'fake_fake_probability': logits[1].detach().sigmoid().item(),
        'real_latent': real_parts['latent'].detach().item(),
        'real_pixel': real_parts['pixel'].detach().item(),
        'real_lpips': real_parts['lpips'].detach().item(),
        'fake_latent': fake_parts['latent'].detach().item(),
        'fake_pixel': fake_parts['pixel'].detach().item(),
        'fake_lpips': fake_parts['lpips'].detach().item(),
        'real_guidance_gradient_rms': real_result['mean_guidance_gradient_rms'].item(),
        'fake_guidance_gradient_rms': fake_result['mean_guidance_gradient_rms'].item(),
        'guided_timesteps': real_result['guided_timesteps'],
    }
    return loss, metrics


def audit(sd, critic, perceptual, conditioning, batch, config, output):
    critic.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    real = batch['real'].to(sd.device)
    fake = batch['fake'].to(sd.device)
    noise = make_noise(sd, config['noise_seed_start'] - 1)
    kwargs = {
        'conditioning': conditioning, 'noise': noise,
        'maximum_timestep': config['maximum_noise_timestep'],
        'differentiable_steps': config['differentiable_guidance_steps'],
        'strength': config['guidance_strength'], 'rms_clip': config['gradient_rms_clip'],
        'checkpointing': True,
    }
    real_result = reconstruct_with_real_guidance(sd, critic, real, **kwargs)
    fake_result = reconstruct_with_real_guidance(sd, critic, fake, **kwargs)
    real_parts = reconstruction_components(real_result, perceptual, config['latent_channel_std'])
    fake_parts = reconstruction_components(fake_result, perceptual, config['latent_channel_std'])
    real_score, fake_score, ranking = reconstruction_objective(
        real_parts, fake_parts, config['reconstruction_weights'], config['ranking_margin'])
    # This excludes ordinary BCE: a nonzero gradient proves the second-order path reaches the critic.
    reconstruction_loss = config['real_preservation_weight'] * real_score + config['ranking_weight'] * ranking
    reconstruction_loss.backward()
    gradients = [parameter.grad for parameter in critic.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError('Reconstruction objective did not reach critic parameters.')
    gradient_norm = torch.stack([gradient.float().norm() for gradient in gradients]).norm()
    if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError('Reconstruction objective did not reach critic parameters.')
    if any(parameter.grad is not None for model in sd.models() for parameter in model.parameters()):
        raise RuntimeError('A frozen SD parameter received a gradient.')
    if any(parameter.grad is not None for parameter in perceptual.parameters()):
        raise RuntimeError('A frozen LPIPS parameter received a gradient.')
    report = {
        'passed': True, 'second_order_critic_gradient_norm': gradient_norm.item(),
        'real_reconstruction': real_score.detach().item(),
        'fake_reconstruction': fake_score.detach().item(),
        'ranking_loss': ranking.detach().item(),
        'low_noise_timesteps': real_result['timesteps'],
        'differentiable_guided_timesteps': real_result['guided_timesteps'],
        'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'seconds': time.monotonic() - started,
    }
    write_json(report, output / 'gradient_audit.json')
    critic.zero_grad(set_to_none=True)
    del real_result, fake_result, real_parts, fake_parts, reconstruction_loss
    torch.cuda.empty_cache()
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/resnet50_reconstruction_guided_sd14.yaml')
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--max-pairs', type=int)
    parser.add_argument('--data-root')
    parser.add_argument('--model-path')
    parser.add_argument('--pretrained')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    for name in ('data_root', 'model_path', 'pretrained', 'output'):
        override = getattr(args, name)
        if override:
            config[name] = override
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16 CUDA GPU is required.')
    if config['precision'] != 'bf16' or config['batch_size_pairs'] != 1:
        raise ValueError('The second-order path requires BF16 and one pair per forward pass.')
    if config['differentiable_guidance_steps'] < 1:
        raise ValueError('At least one differentiable guidance step is required.')
    seed_everything(config['seed'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    latest = output / 'latest.pt'
    if args.resume == 'auto':
        args.resume = str(latest)
    if latest.exists() and not args.resume and not args.audit_only:
        raise FileExistsError(f'Existing run found at {output}; pass --resume.')

    records = official_records(config['data_root'], 'train')
    real_records = [row for row in records if row['label'] == 0]
    fake_records = [row for row in records if row['label'] == 1]
    expected = int(config['expected_per_class'])
    if len(real_records) != expected or len(fake_records) != expected:
        raise ValueError(f'Expected {expected} images per class, found {len(real_records)} and {len(fake_records)}.')
    generator = torch.Generator().manual_seed(config['seed'])
    order = torch.randperm(expected, generator=generator).tolist()
    training_pairs = min(args.max_pairs or config['training_pairs'], expected)
    order = order[:training_pairs]
    pairs = [{'pair_position': position, 'source_index': index,
              'real_path': real_records[index]['path'], 'fake_path': fake_records[index]['path']}
             for position, index in enumerate(order)]
    manifest_text, manifest_hash = manifest(pairs)
    manifest_path = output / 'pairs.jsonl'
    if manifest_path.exists() and manifest_path.read_text() != manifest_text:
        raise ValueError('Existing pair manifest differs.')
    manifest_path.write_text(manifest_text)
    (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

    checkpoint_state = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint_state:
        if checkpoint_state['configuration'] != config or checkpoint_state['pair_manifest_sha256'] != manifest_hash:
            raise ValueError('Resume configuration or pair manifest differs.')
    critic = build_critic(None if checkpoint_state else config['pretrained']).to(device)
    if checkpoint_state:
        critic.load_state_dict(checkpoint_state['model'], strict=True)
    critic.eval().requires_grad_(True)
    critic.apply(freeze_batchnorm)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'], betas=tuple(config['betas']),
                                  eps=config['epsilon'])
    total_updates = math.ceil(training_pairs / config['gradient_accumulation_steps'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=config['minimum_learning_rate'])
    cursor, optimizer_steps, history = 0, 0, []
    if checkpoint_state:
        optimizer.load_state_dict(checkpoint_state['optimizer'])
        scheduler.load_state_dict(checkpoint_state['scheduler'])
        cursor = int(checkpoint_state['cursor'])
        optimizer_steps = int(checkpoint_state['optimizer_steps'])
        history = checkpoint_state.get('history', [])
        restore_rng(checkpoint_state['rng_state'])
    del checkpoint_state

    print('Loading frozen SD 1.4 and LPIPS...', flush=True)
    sd = DifferentiableSD(config['model_path'], dtype=torch.bfloat16,
                          steps=config['ddim_steps'], guidance=1.0, size=config['resolution'])
    import pyiqa
    perceptual = pyiqa.create_metric(
        'lpips', as_loss=True, device=device, net=config['lpips_backbone']).eval().requires_grad_(False)
    with torch.no_grad():
        conditioning = encode_plain(sd.text_encoder, sd.tokenizer, config['conditioning_prompt'])
    write_json({
        'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'training_pairs': training_pairs,
        'optimizer_updates': total_updates, 'pair_manifest_sha256': manifest_hash,
        'packages': {name: version(name) for name in ('torchvision', 'diffusers', 'transformers',
                                                       'pyiqa', 'swanlab')},
    }, output / 'environment.json')

    dataset = ReconstructionPairDataset(
        config['data_root'], real_records, fake_records, config['resolution'], config['decode_attempts'])
    selected = Subset(dataset, order[cursor:])
    loader = DataLoader(selected, batch_size=1, shuffle=False, num_workers=config['workers'],
                        pin_memory=True, persistent_workers=config['workers'] > 0)
    if (config.get('run_gradient_audit', True) and not args.resume
            and not (output / 'gradient_audit.json').exists()):
        audit(sd, critic, perceptual, conditioning, next(iter(loader)), config, output)
    if args.audit_only:
        return

    tracker = None
    if config.get('swanlab'):
        import swanlab
        tracking_path = output / 'swanlab_run.json'
        tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else {'id': uuid.uuid4().hex[:16]}
        tracker = swanlab.init(project=config['swanlab']['project'],
                               experiment_name=config['swanlab']['experiment_name'],
                               config=config, mode='cloud', public=False,
                               logdir=str(output / 'swanlab'), id=tracking['id'], resume='allow')
        tracking['url'] = tracker.url
        write_json(tracking, tracking_path)

    metrics_path = output / 'metrics.jsonl'
    metrics_path.write_text(''.join(json.dumps(row) + '\n' for row in history))
    optimizer.zero_grad(set_to_none=True)
    accumulated = []
    started = time.monotonic()
    for batch in loader:
        position = cursor
        real = batch['real'].to(device, non_blocking=True)
        fake = batch['fake'].to(device, non_blocking=True)
        loss, metrics = pair_objective(
            sd, critic, perceptual, conditioning, real, fake, config,
            config['noise_seed_start'] + position)
        (loss / config['gradient_accumulation_steps']).backward()
        accumulated.append(metrics)
        cursor += 1
        update = len(accumulated) == config['gradient_accumulation_steps'] or cursor == training_pairs
        if not update:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            critic.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        keys = [key for key, value in accumulated[0].items() if isinstance(value, (int, float))]
        row = {key: sum(item[key] for item in accumulated) / len(accumulated) for key in keys}
        row.update({'optimizer_step': optimizer_steps, 'completed_pairs': cursor,
                    'accumulated_pairs': len(accumulated), 'gradient_norm_before_clip': gradient_norm.item(),
                    'learning_rate': learning_rate, 'next_learning_rate': optimizer.param_groups[0]['lr'],
                    'elapsed_seconds': time.monotonic() - started})
        history.append(row)
        with metrics_path.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if tracker:
            tracker.log({f'train/{key}': value for key, value in row.items()
                         if isinstance(value, (int, float)) and key != 'optimizer_step'}, step=optimizer_steps)
        accumulated = []
        if optimizer_steps % config['checkpoint_every_updates'] == 0 or cursor == training_pairs:
            save_state(latest, critic, optimizer, scheduler, cursor, optimizer_steps,
                       config, manifest_hash, history)

    if cursor == training_pairs:
        final = output / 'final.pt'
        save_state(final, critic, optimizer, scheduler, cursor, optimizer_steps,
                   config, manifest_hash, history)
        write_json({'complete': True, 'pairs': cursor, 'optimizer_steps': optimizer_steps,
                    'final_learning_rate': optimizer.param_groups[0]['lr'],
                    'checkpoint': str(final.resolve())}, output / 'summary.json')
    if tracker:
        swanlab.finish()


if __name__ == '__main__':
    main()
