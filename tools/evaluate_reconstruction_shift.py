"""Measure how real-target classifier guidance shifts real-image reconstructions."""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.utils import make_grid

from aigi_detection.datasets import official_records
from aigi_detection.losses.losses import critic_logits
from aigi_detection.models.backbones import build_critic
from aigi_detection.models.modules.differentiable_sd import DifferentiableSD
from aigi_detection.models.modules.guided_reconstruction import reconstruct_with_real_guidance
from aigi_detection.models.modules.soft_prompt import encode_plain


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def noise_for(sd, seed):
    generator = torch.Generator(device=sd.device).manual_seed(seed)
    return torch.randn((1, sd.unet.config.in_channels, sd.size // 8, sd.size // 8),
                       generator=generator, device=sd.device, dtype=torch.float32)


def describe(rows):
    summary = {'count': len(rows)}
    for key in rows[0]:
        if key in {'index', 'path', 'noise_seed'}:
            continue
        values = [float(row[key]) for row in rows]
        ordered = sorted(values)
        summary[key] = {
            'mean': statistics.fmean(values),
            'std': statistics.pstdev(values),
            'median': statistics.median(values),
            'p95': ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
        }
    summary['classified_fake_fraction'] = {
        name: statistics.fmean(float(row[f'{name}_fake_probability'] >= .5) for row in rows)
        for name in ('original', 'unguided', 'guided')
    }
    for metric in ('pixel_l1', 'lpips'):
        increments = [row[f'{metric}_guided_vs_source'] - row[f'{metric}_unguided_vs_source']
                      for row in rows]
        baseline = statistics.fmean(row[f'{metric}_unguided_vs_source'] for row in rows)
        summary[f'{metric}_source_distance_increment'] = {
            'mean': statistics.fmean(increments),
            'median': statistics.median(increments),
            'positive_fraction': statistics.fmean(value > 0 for value in increments),
            'relative_to_unguided_mean': statistics.fmean(increments) / baseline,
        }
    return summary


def save_grid(examples, path):
    # Columns are examples; rows are source, unguided reconstruction, real-guided reconstruction.
    tensor = torch.cat([torch.cat([example[row] for example in examples])
                        for row in ('source', 'unguided', 'guided')])
    grid = make_grid(tensor, nrow=len(examples), padding=4, pad_value=1)
    image = T.ToPILImage()(grid)
    header = 34
    canvas = Image.new('RGB', (image.width, image.height + header), 'white')
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((8, 9), 'Rows: source | unguided | detector-guided to real (label 0)', fill='black')
    canvas.save(path, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/reconstruction_shift_online40k_real100.yaml')
    parser.add_argument('--count', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.count is not None:
        config['sample_count'] = args.count
    if args.output:
        config['output'] = args.output
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16 CUDA GPU is required.')
    torch.manual_seed(config['sample_seed'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)

    all_records = official_records(config['data_root'], config['split'])
    class_label = {'nature': 0, 'ai': 1}[config['class_name']]
    candidates = [row for row in all_records if row['label'] == class_label]
    candidates.sort(key=lambda row: row['path'])
    random.Random(config['sample_seed']).shuffle(candidates)
    selected = candidates[:config['sample_count']]
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in selected))
    (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

    checkpoint_path = Path(config['critic_checkpoint'])
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if state.get('label_mapping') != {'nature': 0, 'ai': 1}:
        raise ValueError('Critic checkpoint does not use nature=0, ai=1.')
    critic = build_critic().to(device)
    critic.load_state_dict(state['model'], strict=True)
    critic.eval().requires_grad_(False)
    del state
    sd = DifferentiableSD(config['model_path'], dtype=torch.bfloat16,
                          steps=config['ddim_steps'], guidance=1.0,
                          size=config['resolution'])
    import pyiqa
    perceptual = pyiqa.create_metric(
        'lpips', as_loss=True, device=device, net=config['lpips_backbone']).eval().requires_grad_(False)
    with torch.no_grad():
        conditioning = encode_plain(sd.text_encoder, sd.tokenizer, config['conditioning_prompt'])
    transform = T.Compose((T.Resize(config['resolution'], interpolation=T.InterpolationMode.BICUBIC,
                                     antialias=True), T.CenterCrop(config['resolution']), T.ToTensor()))

    rows, examples, failures = [], [], []
    used_timesteps, used_guided_timesteps = [], []
    started = time.monotonic()
    for source_index, record in enumerate(selected):
        try:
            with Image.open(Path(config['data_root']) / record['path']) as image:
                source = transform(image.convert('RGB')).unsqueeze(0).to(device)
        except (OSError, ValueError, SyntaxError) as error:
            failures.append({'path': record['path'], 'error': f'{type(error).__name__}: {error}'})
            continue
        seed = config['noise_seed_start'] + source_index
        result = reconstruct_with_real_guidance(
            sd, critic, source, conditioning, noise_for(sd, seed),
            maximum_timestep=config['maximum_noise_timestep'],
            differentiable_steps=config['guided_steps'], strength=config['guidance_strength'],
            rms_clip=config['gradient_rms_clip'], checkpointing=False, second_order=False)
        used_timesteps = result['timesteps']
        used_guided_timesteps = result['guided_timesteps']
        baseline, guided = result['baseline_image'], result['guided_image']
        with torch.no_grad():
            probabilities = critic_logits(critic, torch.cat((source, baseline, guided))).sigmoid()
            lpips_shift = perceptual(guided.float(), baseline.float()).mean()
            lpips_base = perceptual(baseline.float(), source.float()).mean()
            lpips_guided = perceptual(guided.float(), source.float()).mean()
        row = {
            'index': source_index, 'path': record['path'], 'noise_seed': seed,
            'original_fake_probability': probabilities[0].item(),
            'unguided_fake_probability': probabilities[1].item(),
            'guided_fake_probability': probabilities[2].item(),
            'fake_probability_change': (probabilities[2] - probabilities[1]).item(),
            'latent_mse_guided_vs_unguided': (result['guided_latent'] - result['baseline_latent']).square().mean().item(),
            'pixel_l1_guided_vs_unguided': (guided - baseline).abs().mean().item(),
            'lpips_guided_vs_unguided': lpips_shift.item(),
            'pixel_l1_unguided_vs_source': (baseline - source).abs().mean().item(),
            'pixel_l1_guided_vs_source': (guided - source).abs().mean().item(),
            'lpips_unguided_vs_source': lpips_base.item(),
            'lpips_guided_vs_source': lpips_guided.item(),
            'mean_guidance_gradient_rms': result['mean_guidance_gradient_rms'].item(),
        }
        rows.append(row)
        if len(examples) < config['grid_examples']:
            examples.append({'source': source.cpu(), 'unguided': baseline.cpu(), 'guided': guided.cpu()})
        print(json.dumps({'done': len(rows), 'requested': config['sample_count'], **row}), flush=True)
        del result, source, baseline, guided

    if not rows:
        raise RuntimeError('No readable images were evaluated.')
    (output / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summary = describe(rows)
    summary.update({
        'failed_images': failures, 'elapsed_seconds': time.monotonic() - started,
        'guided_target': {'label': 0, 'class': 'real/nature'},
        'critic': {'path': str(checkpoint_path.resolve()), 'sha256': digest(checkpoint_path)},
        'timesteps': used_timesteps,
        'guided_timesteps': used_guided_timesteps,
    })
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    save_grid(examples, output / 'matched_grid.jpg')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
